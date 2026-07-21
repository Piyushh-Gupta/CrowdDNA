"""
CrowdFlow DNA — Graph Dataset
==============================
Module: crowdflow_dna/graph/graph_dataset.py

Dataset layer bridging synthetic trajectory JSON files to PyTorch Geometric.
Reads manifest.json, lazily loads JSON records, and converts them to
PyG Data sequences using GraphBuilder.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from crowdflow_dna.graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    """PyTorch Geometric Dataset for loading synthetic crowd trajectories.

    Lazily loads JSON trajectory files specified in a manifest and yields
    sequences of PyG Data objects representing per-frame proximity graphs.

    To prevent unnecessary I/O, each `get(idx)` call loads one complete
    JSON sequence and returns a list of T Data objects, matching the
    expected input format for downstream temporal encoders (e.g. GRU).
    """

    LABEL_MAP = {
        "Safe": 0,
        "Congesting": 1,
        "Critical": 2,
    }

    def __init__(
        self,
        root: str,
        proximity_radius: float = 0.15,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        """Initialises the GraphDataset.

        Args:
            root: Directory containing manifest.json and trajectory files.
            proximity_radius: Radius for edge creation in GraphBuilder.
            transform: Optional PyG transform applied to each Data object.
            pre_transform: Optional PyG pre-transform.
        """
        self.data_dir = root
        self.proximity_radius = proximity_radius
        self.builder = GraphBuilder(proximity_radius)
        self.manifest_path = os.path.join(self.data_dir, "manifest.json")

        self._manifest: list[dict[str, Any]] = []
        self._load_manifest()

        # Pass root=None to prevent PyG from creating raw/processed folders
        # and attempting file processing logic. We handle loading natively.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def _load_manifest(self) -> None:
        """Loads and validates manifest.json."""
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in manifest {self.manifest_path}: {exc}"
            ) from exc

        if not isinstance(self._manifest, list):
            raise ValueError(f"Manifest at {self.manifest_path} must be a JSON array.")

    def len(self) -> int:
        """Returns the total number of trajectory sequences."""
        return len(self._manifest)

    def get(self, idx: int) -> List[Data]:
        """Loads a single trajectory sequence and converts it to graphs.

        Args:
            idx: Index of the sequence in the manifest.

        Returns:
            A list of PyG Data objects, one per timestep in the sequence.

        Raises:
            FileNotFoundError: If the trajectory JSON file is missing.
            ValueError: If the JSON is invalid or the schema is corrupted.
        """
        entry = self._manifest[idx]
        file_name = entry.get("file_path")
        if not file_name:
            raise ValueError(f"Manifest entry {idx} is missing 'file_path'")

        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Trajectory JSON missing: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc

        # Validate schema
        for key in ["positions", "velocities", "frame_labels"]:
            if key not in record:
                raise ValueError(f"Missing required key '{key}' in {file_path}")

        positions = np.array(record["positions"], dtype=np.float32)
        velocities = np.array(record["velocities"], dtype=np.float32)
        labels = record["frame_labels"]

        num_timesteps = len(labels)
        if positions.shape[0] != num_timesteps or velocities.shape[0] != num_timesteps:
            raise ValueError(
                f"Timestep mismatch in {file_path}: labels={num_timesteps}, "
                f"positions={positions.shape[0]}, velocities={velocities.shape[0]}"
            )

        graphs: List[Data] = []
        for t in range(num_timesteps):
            pos_t = positions[t]
            vel_t = velocities[t]

            # Build PyG Data object
            data = self.builder.build(pos_t, vel_t)

            # Preserve label
            label_str = labels[t]
            if label_str not in self.LABEL_MAP:
                raise ValueError(f"Unknown label '{label_str}' in {file_path}")

            # Store as a 1D tensor
            data.y = torch.tensor([self.LABEL_MAP[label_str]], dtype=torch.long)
            graphs.append(data)

        return graphs
