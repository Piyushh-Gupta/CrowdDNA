"""
CrowdFlow DNA — Robustness Tests
================================
Module: tests/test_robustness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from training.robustness.context import EvaluationContext
from training.robustness.datasets import DatasetProvider
from training.robustness.perturbations import (
    add_gaussian_noise,
    drop_frames,
    drop_nodes,
)
from training.robustness.pipeline import EvaluationPipeline
import numpy as np

# Mock Dataset
class MockDatasetProvider(DatasetProvider):
    def __init__(self, num_sequences: int = 5):
        self.num_sequences = num_sequences
        self.seq_len = 5
        self.num_nodes = 10
        self.in_channels = 5
        
    def __iter__(self):
        for i in range(self.num_sequences):
            sequence = []
            for j in range(self.seq_len):
                x = torch.ones((self.num_nodes, self.in_channels), dtype=torch.float32)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 4), dtype=torch.float32)
                sequence.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            label = torch.tensor(0, dtype=torch.long)
            yield sequence, label
            
    def __len__(self) -> int:
        return self.num_sequences


class MockInferenceRuntime:
    def load_model(self, path: Path) -> None:
        pass
        
    def predict(self, x, edge_index, edge_attr, batch, seq_lengths):
        from crowdflow_dna.inference.runtime import InferenceResult
        return InferenceResult(
            predicted_class=0,
            probabilities=np.array([0.8, 0.1, 0.1]),
            confidence=0.8,
            backend="mock",
            inference_time_ms=1.5,
            model_format="mock"
        )

def test_drop_nodes():
    seq = [Data(x=torch.ones((10, 5)), edge_index=torch.zeros((2, 0)), edge_attr=torch.zeros((0, 4)))]
    gen = torch.Generator().manual_seed(42)
    # Drop all nodes
    perturbed = drop_nodes(seq, 1.0, gen)
    assert perturbed[0].num_nodes == 0
    
    # Drop no nodes
    perturbed = drop_nodes(seq, 0.0, gen)
    assert perturbed[0].num_nodes == 10

def test_add_gaussian_noise():
    seq = [Data(x=torch.zeros((10, 5)), edge_index=torch.zeros((2, 0)), edge_attr=torch.zeros((0, 4)))]
    gen = torch.Generator().manual_seed(42)
    perturbed = add_gaussian_noise(seq, 1.0, gen)
    assert perturbed[0].x.sum().item() != 0.0

def test_drop_frames():
    seq = [Data(x=torch.ones((10, 5)), edge_index=torch.zeros((2, 0)), edge_attr=torch.zeros((0, 4)))]
    gen = torch.Generator().manual_seed(42)
    perturbed = drop_frames(seq, 1.0, gen)
    assert perturbed[0].num_nodes == 0

def test_pipeline_execution(tmp_path, monkeypatch):
    context = EvaluationContext(
        deployment_model_path=Path("dummy.pt"),
        dataset_provider=MockDatasetProvider(2),
        random_seed=42,
        protocol_name="clean",
        device=torch.device("cpu"),
        output_directory=tmp_path,
    )
    
    # Mock InferenceRuntime to avoid needing a real artifact
    import training.robustness.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "InferenceRuntime", MockInferenceRuntime)
    
    pipeline = EvaluationPipeline(context)
    pipeline.run()
    
    assert (tmp_path / "clean" / "results.json").exists()
    assert (tmp_path / "clean" / "results.csv").exists()
    assert (tmp_path / "clean" / "report.md").exists()

def test_robustness_protocol_execution(tmp_path, monkeypatch):
    context = EvaluationContext(
        deployment_model_path=Path("dummy.pt"),
        dataset_provider=MockDatasetProvider(2),
        random_seed=42,
        protocol_name="robustness",
        device=torch.device("cpu"),
        output_directory=tmp_path,
        config={"scenarios": {"missing_detections": {"drop_prob": 0.5}}}
    )
    
    import training.robustness.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "InferenceRuntime", MockInferenceRuntime)
    
    # Mock plotting to avoid generating PNGs during fast tests
    import training.robustness.report as report_mod
    monkeypatch.setattr(report_mod.ReportBuilder, "_generate_plots", lambda self: None)
    
    pipeline = EvaluationPipeline(context)
    pipeline.run()
    
    out_dir = tmp_path / "robustness"
    assert (out_dir / "results.json").exists()
    with open(out_dir / "results.json", "r") as f:
        data = json.load(f)
        assert "baseline" in data["results"]
        assert "missing_detections" in data["results"]
