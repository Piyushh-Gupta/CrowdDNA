"""
CrowdFlow DNA — Sequence Graph Dataset
======================================
Module: crowdflow_dna/graph/sequence_dataset.py
Owner: Piyush Gupta (AI & Data Lead)

Dataset layer for sequence-level learning. Reads synthetic trajectory JSON files
exactly once per trajectory and yields the entire sequence of proximity graphs
along with a trajectory-level risk label.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from crowdflow_dna.graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, int] = {
    "Safe": 0,
    "Congesting": 1,
    "Critical": 2,
}


@dataclass(frozen=True)
class SequenceSample:
    """Return type for one trajectory from SequenceGraphDataset."""
    sequence_id: str
    graphs: list[Data]
    label: torch.Tensor


class SequenceGraphDataset(Dataset):
    """PyTorch Geometric Dataset for loading complete synthetic crowd trajectories.

    Lazily loads JSON trajectory files specified in a manifest. Unlike GraphDataset,
    which yields individual frames, this dataset yields an entire trajectory sequence
    per index.
    """

    def __init__(
        self,
        root: str,
        proximity_radius: float = 2.0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        """Initialises the SequenceGraphDataset.

        Args:
            root: Directory containing manifest.json and trajectory files.
            proximity_radius: Physical radius (metres) for edge creation.
            transform: Optional PyG transform applied to each Data object.
            pre_transform: Optional PyG pre-transform.
        """
        # Pass root=None to prevent PyG from creating raw/processed folders
        # and attempting file processing logic. We handle loading natively.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        
        self.data_dir = root
        self.proximity_radius = proximity_radius
        self.builder = GraphBuilder(proximity_radius)
        self.manifest_path = os.path.join(self.data_dir, "manifest.json")

        self._manifest: list[dict[str, Any]] = []
        self._load_manifest()

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
        """Returns the total number of trajectories (sequences) in the dataset."""
        return len(self._manifest)

    def get(self, idx: int) -> SequenceSample:
        """Loads a complete trajectory for a specific sequence index.

        Args:
            idx: Index of the trajectory sequence in the manifest.

        Returns:
            A SequenceSample exposing:
            - sequence_id: Unique identifier for the trajectory.
            - graphs: List[Data] representing the sequence of frames.
            - label: torch.Tensor containing the global trajectory label.

        Raises:
            FileNotFoundError: If the trajectory JSON file is missing.
            ValueError: If the JSON is invalid or the schema is corrupted.
        """
        entry = self._manifest[idx]
        
        file_name = entry.get("file_path")
        if not file_name:
            raise ValueError(f"Manifest entry {idx} is missing 'file_path'")

        file_path = os.path.join(self.data_dir, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Trajectory JSON missing: {file_path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc

        # Validate schema
        for key in ["positions", "velocities", "frame_labels"]:
            if key not in record:
                raise ValueError(f"Missing required key '{key}' in {file_path}")

        positions = np.array(record["positions"], dtype=np.float32)
        velocities = np.array(record["velocities"], dtype=np.float32)
        frame_labels = record["frame_labels"]

        num_timesteps = len(frame_labels)
        if positions.shape[0] != num_timesteps or velocities.shape[0] != num_timesteps:
            raise ValueError(
                f"Timestep mismatch in {file_path}: labels={num_timesteps}, "
                f"positions={positions.shape[0]}, velocities={velocities.shape[0]}"
            )

        # 1. Determine sequence label (trajectory-level target)
        # ADR-007: Use the 'risk_class' metadata provided by the simulator. 
        # If missing, fallback to the maximum risk level among all frames.
        risk_class = record.get("risk_class")
        if risk_class in LABEL_MAP:
            seq_label_idx = LABEL_MAP[risk_class]
        else:
            # Fallback aggregation: take the maximum risk seen in any frame
            frame_label_indices = [LABEL_MAP[lbl] for lbl in frame_labels if lbl in LABEL_MAP]
            if not frame_label_indices:
                raise ValueError(f"No valid frame labels found in {file_path}")
            seq_label_idx = max(frame_label_indices)
            
        sequence_label = torch.tensor(seq_label_idx, dtype=torch.long)
        
        sequence_id = record.get("sequence_id")
        if sequence_id is None:
            sequence_id = f"seq_{idx}"
            logger.warning("Trajectory at idx=%d has no sequence_id; using default %s.", idx, sequence_id)

        # 2. Build the sequence of PyG Data graphs
        sequence_graphs: list[Data] = []
        for t in range(num_timesteps):
            pos_t = positions[t]
            vel_t = velocities[t]

            # Build PyG Data object
            data = self.builder.build(pos_t, vel_t)

            # Preserve frame-level label (for future metrics or multi-task learning)
            label_str = frame_labels[t]
            if label_str not in LABEL_MAP:
                raise ValueError(f"Unknown label '{label_str}' in {file_path}")

            data.y = torch.tensor([LABEL_MAP[label_str]], dtype=torch.long)

            if self.transform is not None:
                data = self.transform(data)

            sequence_graphs.append(data)

        return SequenceSample(
            sequence_id=sequence_id,
            graphs=sequence_graphs,
            label=sequence_label,
        )
