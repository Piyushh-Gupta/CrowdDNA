"""
CrowdFlow DNA — Sequence Buffer
================================
Module: crowdflow_dna/inference/sequence_buffer.py

Converts per-frame ``torch_geometric.data.Data`` graphs produced by
``GraphBuilder`` into the flat-tensor batch that ``InferenceRuntime.predict``
expects. Implements a sliding-window buffer as defined in INTEGRATION_CONTRACT.md
§6.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import NamedTuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Canonical node-feature and edge-feature dimensions.
# These must match the dimensions used during model training.
_NODE_FEATURE_DIM: int = 5
_EDGE_FEATURE_DIM: int = 4

# Shared constants to avoid repeated allocations (issue m4)
_EMPTY_EDGE_INDEX = torch.zeros((2, 0), dtype=torch.long)
_EMPTY_EDGE_ATTR = torch.zeros((0, _EDGE_FEATURE_DIM), dtype=torch.float32)
_EMPTY_NODE_X = torch.zeros((0, _NODE_FEATURE_DIM), dtype=torch.float32)


class TensorBatch(NamedTuple):
    """Flat-tensor representation of a single-sequence batch.

    Attributes:
        x: Node feature matrix, shape ``(total_nodes, 5)``, float32.
        edge_index: COO edge indices, shape ``(2, total_edges)``, int64.
        edge_attr: Edge features, shape ``(total_edges, 4)``, float32.
        batch: Node-to-frame assignment, shape ``(total_nodes,)``, int64.
        seq_lengths: Number of frames in the window, shape ``(1,)``, int64.
    """

    x: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    seq_lengths: Tensor


class SequenceBuffer:
    """Sliding-window buffer that assembles per-frame PyG Data objects
    into the flat tensor batch required by ``InferenceRuntime``.

    The buffer collects exactly ``window_size`` frames. When full it emits a
    ``TensorBatch`` suitable for ``InferenceRuntime.predict()``. On each
    subsequent frame the oldest entry is evicted (sliding window with
    ``stride=1``).

    Frames with no detections (``data.num_nodes == 0``) are accepted without
    error; they contribute a zero-length slice to the concatenated tensors.

    Call :meth:`reset` before processing a new video or when the tracking
    session is interrupted.

    Args:
        window_size: Number of consecutive frames per inference request.
    """

    def __init__(self, window_size: int = 10) -> None:
        """Initialise the buffer.

        Args:
            window_size: Number of consecutive frames forming one inference
                request. Must be ≥ 1.

        Raises:
            ValueError: If ``window_size`` is less than 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        self._window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Configured window size (read-only)."""
        return self._window_size

    @property
    def current_size(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when the buffer holds exactly ``window_size`` frames."""
        return len(self._buffer) == self._window_size

    def push(self, data: object) -> None:
        """Push a single per-frame ``Data`` object into the buffer.

        The buffer automatically evicts the oldest frame once full
        (``deque`` with ``maxlen``).

        Args:
            data: A ``torch_geometric.data.Data`` object from
                ``GraphBuilder.build()``. Must have attributes ``x``,
                ``edge_index``, ``edge_attr``, and ``num_nodes``.
        """
        self._buffer.append(data)

    def assemble(self) -> TensorBatch:
        """Assemble the current window into a ``TensorBatch``.

        Must only be called when :attr:`is_ready` is ``True``.

        Returns:
            A ``TensorBatch`` ready to be unpacked into
            ``InferenceRuntime.predict()``.

        Raises:
            RuntimeError: If the buffer is not yet full.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"SequenceBuffer.assemble() called with only {len(self._buffer)}"
                f" of {self._window_size} frames available."
            )

        x_parts: list[Tensor] = []
        edge_index_parts: list[Tensor] = []
        edge_attr_parts: list[Tensor] = []
        batch_parts: list[Tensor] = []

        cumulative_offset = 0

        for frame_idx, data in enumerate(self._buffer):
            n: int = int(data.num_nodes) if data.num_nodes is not None else 0

            # -- Node features -----------------------------------------
            if n > 0:
                x_parts.append(data.x.float())
            else:
                x_parts.append(_EMPTY_NODE_X)

            # -- Edge indices with global offset -----------------------
            if getattr(data, "edge_index", None) is not None and data.edge_index.shape[1] > 0:
                edge_index_parts.append(data.edge_index + cumulative_offset)
            else:
                edge_index_parts.append(_EMPTY_EDGE_INDEX)

            # -- Edge attributes ---------------------------------------
            if getattr(data, "edge_attr", None) is not None and data.edge_attr.shape[0] > 0:
                edge_attr_parts.append(data.edge_attr.float())
            else:
                edge_attr_parts.append(_EMPTY_EDGE_ATTR)

            # -- Batch vector (node → frame index) ---------------------
            if n > 0:
                batch_parts.append(
                    torch.full((n,), frame_idx, dtype=torch.long)
                )

            cumulative_offset += n

        x = torch.cat(x_parts, dim=0)
        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_attr = torch.cat(edge_attr_parts, dim=0)
        batch = (
            torch.cat(batch_parts, dim=0)
            if batch_parts
            else torch.zeros(0, dtype=torch.long)
        )
        seq_lengths = torch.tensor([self._window_size], dtype=torch.long)

        logger.debug(
            "SequenceBuffer.assemble(): %d frames, %d nodes, %d edges",
            self._window_size,
            x.shape[0],
            edge_index.shape[1],
        )

        return TensorBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            seq_lengths=seq_lengths,
        )

    def reset(self) -> None:
        """Clear all buffered frames.

        Call between videos or on track-session discontinuity.
        """
        self._buffer.clear()
        logger.debug("SequenceBuffer.reset(): buffer cleared.")
