"""
CrowdFlow DNA — Graph Builder
==============================
Module: crowdflow_dna/graph/graph_builder.py
Owner:  Piyush Gupta (AI & Data Lead)

Converts per-frame pedestrian detections into a proximity graph compatible
with PyTorch Geometric (PyG). Each pedestrian becomes a node; edges connect
pairs of pedestrians whose normalised Euclidean distance is within the
configured proximity radius.

The output ``torch_geometric.data.Data`` object is the authoritative input
contract for the downstream GNN encoder.

SRD References:
    §4.6.4 Graph Construction
    §4.6.5 GNN+GRU Model Architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("crowdflow.graph_builder")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single tracked pedestrian detection for one video frame.

    Coordinates are expressed in normalised frame space [0, 1] so that the
    proximity threshold in the configuration is frame-size-independent.

    Attributes:
        track_id: Unique integer identifier assigned by the tracker.
            Stable across frames for the same person.
        x: Normalised horizontal centre coordinate in [0, 1].
        y: Normalised vertical centre coordinate in [0, 1].
        confidence: Detection confidence score in [0, 1].
    """

    track_id: int
    x: float
    y: float
    confidence: float


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Converts a list of per-frame detections into a PyG Data graph.

    One ``GraphBuilder`` instance is created per pipeline run and reused
    across all frames. Its only configurable parameter is
    ``proximity_radius``, read from the ``graph.proximity_radius`` key in
    the YAML configuration.

    Responsibility boundary:
        - Receives a list of ``Detection`` objects for a single frame.
        - Returns a ``torch_geometric.data.Data`` object.
        - Does NOT perform detection, tracking, or model inference.
    """

    def __init__(self, proximity_radius: float) -> None:
        """Initialises the GraphBuilder.

        Args:
            proximity_radius: Maximum normalised Euclidean distance between
                two pedestrian centres for an edge to be created. Must be
                in the range (0, 1] since coordinates are normalised.

        Raises:
            ValueError: If ``proximity_radius`` is not in (0, 1].
        """
        if not (0.0 < proximity_radius <= 1.0):
            raise ValueError(
                f"proximity_radius must be in (0, 1], got {proximity_radius}."
            )
        self._proximity_radius = proximity_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, detections: list[Detection]) -> Any:
        """Builds a PyG Data graph from a list of detections for one frame.

        Node features (``x``) are a ``(N, 3)`` float32 tensor:
        ``[norm_x, norm_y, confidence]`` for each detection.

        Edges are undirected and connect every pair of nodes whose
        normalised Euclidean distance is strictly less than or equal to
        ``proximity_radius``. Self-loops are excluded. Edge indices are
        stored in COO format as a ``(2, E)`` long tensor.

        Edge attributes (``edge_attr``) are a ``(E, 1)`` float32 tensor
        containing the normalised Euclidean distance for each edge.

        When fewer than two detections are present no edges can be formed.
        The function still returns a valid Data object with an empty
        ``edge_index`` of shape ``(2, 0)``.

        Args:
            detections: Tracked pedestrian detections for a single frame.
                May be empty.

        Returns:
            A ``torch_geometric.data.Data`` instance with attributes:
                - ``x``: Node feature matrix, shape ``(N, 3)``, float32.
                - ``edge_index``: COO edge indices, shape ``(2, E)``, long.
                - ``edge_attr``: Edge distances, shape ``(E, 1)``, float32.
                - ``num_nodes``: Integer count of nodes ``N``.
                - ``track_ids``: 1-D long tensor of track IDs, shape ``(N,)``.

        Raises:
            ImportError: If ``torch_geometric`` is not installed.
        """
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GraphBuilder. "
                "Install it with: pip install torch-geometric"
            ) from exc

        n = len(detections)

        node_features, track_ids = self._build_node_features(detections)
        edge_index, edge_attr = self._build_edges(detections)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n,
            track_ids=track_ids,
        )

        logger.debug(
            "Built graph: %d nodes, %d edges (radius=%.3f)",
            n, edge_index.shape[1], self._proximity_radius,
        )
        return graph

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_node_features(
        self, detections: list[Detection]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs node feature and track-ID tensors.

        Args:
            detections: Detections for the current frame.

        Returns:
            Tuple of:
                - ``node_features``: Float32 tensor of shape ``(N, 3)``.
                - ``track_ids``: Long tensor of shape ``(N,)``.
        """
        if not detections:
            return (
                torch.zeros((0, 3), dtype=torch.float32),
                torch.zeros(0, dtype=torch.long),
            )

        features = np.array(
            [[d.x, d.y, d.confidence] for d in detections], dtype=np.float32
        )
        ids = np.array([d.track_id for d in detections], dtype=np.int64)
        return torch.from_numpy(features), torch.from_numpy(ids)

    def _build_edges(
        self, detections: list[Detection]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds COO edge indices and distance edge attributes.

        Computes pairwise Euclidean distances between all detections and
        creates an undirected edge for every pair within
        ``proximity_radius``.  The result includes both directions for
        each edge so that the graph is symmetric (required by most PyG
        message-passing layers).

        Args:
            detections: Detections for the current frame.

        Returns:
            Tuple of:
                - ``edge_index``: Long tensor of shape ``(2, E)``.
                - ``edge_attr``: Float32 tensor of shape ``(E, 1)``
                  containing the normalised Euclidean distance.
        """
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float32)

        n = len(detections)
        if n < 2:
            return empty_index, empty_attr

        coords = np.array([[d.x, d.y] for d in detections], dtype=np.float32)

        # Pairwise distance matrix via broadcasting — shape (N, N)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))             # (N, N)

        # Upper-triangle indices for unique pairs (excludes self-loops)
        row_upper, col_upper = np.triu_indices(n, k=1)
        distances = dist_matrix[row_upper, col_upper]
        mask = distances <= self._proximity_radius

        if not mask.any():
            return empty_index, empty_attr

        src = row_upper[mask]
        dst = col_upper[mask]
        dists = distances[mask].astype(np.float32)

        # Make undirected: add both (src→dst) and (dst→src)
        edge_src = np.concatenate([src, dst])
        edge_dst = np.concatenate([dst, src])
        edge_dists = np.concatenate([dists, dists])

        edge_index = torch.from_numpy(
            np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
        )
        edge_attr = torch.from_numpy(edge_dists[:, np.newaxis])

        return edge_index, edge_attr

    @property
    def proximity_radius(self) -> float:
        """The configured proximity radius (read-only)."""
        return self._proximity_radius
