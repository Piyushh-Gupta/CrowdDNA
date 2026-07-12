"""
CrowdFlow DNA — Graph Builder
==============================
Module: crowdflow_dna/graph/graph_builder.py
Owner:  Piyush Gupta (AI & Data Lead)

Constructs per-frame proximity graphs from normalised position and velocity
arrays produced by the Phase 5 synthetic training pipeline
(``TrajectoryRecord.positions``, ``TrajectoryRecord.velocities``).

Each pedestrian becomes a node; undirected edges connect pairs whose
normalised Euclidean distance is within the configured proximity radius.

Node features  (N, 5): [x, y, vx, vy, speed]
Edge features  (E, 4): [dx, dy, distance, relative_speed]

The output ``torch_geometric.data.Data`` object is the authoritative input
contract for the downstream GNN encoder (Phase 7).

SRD References:
    §3.1  FR-005  Graph edge features
    §4.6.3  Proximity-based graph construction
    §4.6.4  Planned node / edge feature sets
    §4.6.5  GNN+GRU Model Architecture
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("crowdflow.graph_builder")


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Builds per-frame proximity graphs for the training pipeline.

    Accepts raw position and velocity arrays for a single timestep —
    exactly the slices produced by loading a ``TrajectoryRecord`` — and
    returns a ``torch_geometric.data.Data`` object ready for the GNN
    encoder.

    Responsibility boundary:
        - Receives ``positions (N, 2)`` and ``velocities (N, 2)``
          in normalised frame coordinates for a single frame.
        - Returns a ``torch_geometric.data.Data`` object.
        - Does NOT perform detection, tracking, serialisation, or inference.
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

    def build(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Any:
        """Builds a PyG Data graph from per-frame position and velocity arrays.

        Node features (``x``) are a ``(N, 5)`` float32 tensor:
        ``[x, y, vx, vy, speed]`` for each agent.

        Edges are undirected and connect every pair of agents whose
        normalised Euclidean distance is within ``proximity_radius``.
        Self-loops are excluded. Edge indices are stored in COO format as
        a ``(2, E)`` long tensor.

        Edge attributes (``edge_attr``) are a ``(E, 4)`` float32 tensor:
        ``[dx, dy, distance, relative_speed]`` for each edge, where both
        directed copies of a pair carry the same spatial magnitude but
        opposite ``[dx, dy]`` components.

        When fewer than two agents are present, no edges can be formed.
        The function still returns a valid Data object with empty
        ``edge_index`` of shape ``(2, 0)``.

        Args:
            positions: Agent positions for a single frame.
                Shape ``(N, 2)``, dtype float32-compatible, values in [0, 1].
            velocities: Agent velocities for the same frame.
                Shape ``(N, 2)``, dtype float32-compatible.

        Returns:
            A ``torch_geometric.data.Data`` instance with attributes:
                - ``x``: Node feature matrix, shape ``(N, 5)``, float32.
                - ``edge_index``: COO edge indices, shape ``(2, E)``, long.
                - ``edge_attr``: Edge features, shape ``(E, 4)``, float32.
                - ``num_nodes``: Integer count of nodes ``N``.

        Raises:
            ValueError: If ``positions`` and ``velocities`` differ in shape,
                or do not have exactly 2 columns.
            ImportError: If ``torch_geometric`` is not installed.
        """
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GraphBuilder. "
                "Install it with: pip install torch-geometric"
            ) from exc

        positions = np.asarray(positions, dtype=np.float32)
        velocities = np.asarray(velocities, dtype=np.float32)
        self._validate_inputs(positions, velocities)

        n = positions.shape[0]
        node_features = self._build_node_features(positions, velocities)
        edge_index, edge_attr = self._build_edges(positions, velocities)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n,
        )

        logger.debug(
            "Built graph: %d nodes, %d edges (radius=%.3f)",
            n, edge_index.shape[1], self._proximity_radius,
        )
        return graph

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """Validates that position and velocity arrays are compatible.

        Args:
            positions: Position array to validate.
            velocities: Velocity array to validate.

        Raises:
            ValueError: If shapes are incompatible or columns != 2.
        """
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"positions must have shape (N, 2), got {positions.shape}."
            )
        if velocities.ndim != 2 or velocities.shape[1] != 2:
            raise ValueError(
                f"velocities must have shape (N, 2), got {velocities.shape}."
            )
        if positions.shape[0] != velocities.shape[0]:
            raise ValueError(
                f"positions and velocities must have the same number of agents. "
                f"Got {positions.shape[0]} vs {velocities.shape[0]}."
            )

    @staticmethod
    def _compute_speed(velocities: np.ndarray) -> np.ndarray:
        """Computes the per-agent L2 speed from a velocity array.

        Args:
            velocities: Float32 array of shape ``(N, 2)``.

        Returns:
            Float32 array of shape ``(N,)`` containing the speed
            (Euclidean norm of the velocity vector) for each agent.
        """
        return np.linalg.norm(velocities, axis=1).astype(np.float32)

    @staticmethod
    def _build_node_features(
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> torch.Tensor:
        """Constructs the (N, 5) node feature tensor.

        Features per node: [x, y, vx, vy, speed].
        Speed is computed via :meth:`_compute_speed`.

        Args:
            positions: Float32 array of shape ``(N, 2)``.
            velocities: Float32 array of shape ``(N, 2)``.

        Returns:
            Float32 tensor of shape ``(N, 5)``.
        """
        if positions.shape[0] == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        speed = GraphBuilder._compute_speed(velocities)[:, np.newaxis]
        features = np.concatenate([positions, velocities, speed], axis=1)
        return torch.from_numpy(features)

    def _build_edges(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds COO edge indices and (E, 4) edge attribute tensor.

        Edge features per directed edge (src → dst):
            [dx, dy, distance, relative_speed]

        ``dx`` and ``dy`` are (dst - src) position differences, giving each
        directed edge a unique sign so message-passing can encode directionality.
        ``distance`` is the symmetric Euclidean distance.
        ``relative_speed`` is the L2 norm of (vel_dst - vel_src), symmetric.

        Args:
            positions: Float32 array of shape ``(N, 2)``.
            velocities: Float32 array of shape ``(N, 2)``.

        Returns:
            Tuple of:
                - ``edge_index``: Long tensor of shape ``(2, E)``.
                - ``edge_attr``: Float32 tensor of shape ``(E, 4)``.
        """
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 4), dtype=torch.float32)

        n = positions.shape[0]
        if n < 2:
            return empty_index, empty_attr

        # Pairwise position differences: diff[i,j] = pos[j] - pos[i]
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))                   # (N, N)

        # Pairwise relative velocity differences: rel_vel[i,j] = vel[j] - vel[i]
        vel_diff = velocities[np.newaxis, :, :] - velocities[:, np.newaxis, :]  # (N, N, 2)
        rel_speed_matrix = np.sqrt((vel_diff ** 2).sum(axis=-1))                # (N, N)

        # Upper-triangle indices for unique pairs (excludes self-loops)
        row_upper, col_upper = np.triu_indices(n, k=1)
        distances = dist_matrix[row_upper, col_upper]
        mask = distances <= self._proximity_radius

        if not mask.any():
            return empty_index, empty_attr

        src = row_upper[mask]
        dst = col_upper[mask]
        dists = distances[mask].astype(np.float32)

        # [dx, dy] for src→dst
        dx_fwd = diff[src, dst, 0].astype(np.float32)
        dy_fwd = diff[src, dst, 1].astype(np.float32)

        # relative_speed is symmetric
        rel_speeds = rel_speed_matrix[src, dst].astype(np.float32)

        # Forward edge (src → dst): [dx, dy, dist, rel_speed]
        fwd_attr = np.stack([dx_fwd, dy_fwd, dists, rel_speeds], axis=1)
        # Reverse edge (dst → src): [-dx, -dy, dist, rel_speed]
        rev_attr = np.stack([-dx_fwd, -dy_fwd, dists, rel_speeds], axis=1)

        edge_src = np.concatenate([src, dst])
        edge_dst = np.concatenate([dst, src])
        attr = np.concatenate([fwd_attr, rev_attr], axis=0).astype(np.float32)

        edge_index = torch.from_numpy(
            np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
        )
        edge_attr = torch.from_numpy(attr)

        return edge_index, edge_attr

    @property
    def proximity_radius(self) -> float:
        """The configured proximity radius (read-only)."""
        return self._proximity_radius
