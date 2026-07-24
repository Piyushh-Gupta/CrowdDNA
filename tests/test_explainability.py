"""
Tests for Explainability Framework.
"""
import pytest
import torch
import os
import shutil
import logging
from torch_geometric.data import Data

from crowdflow_dna.explainability.registry import ExplainerRegistry, ExplainerCategory
from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph
from crowdflow_dna.explainability.pipeline import ExplanationPipeline
from crowdflow_dna.explainability.cache import CacheManager
from crowdflow_dna.inference.runtime import InferenceRuntime

import crowdflow_dna.explainability.explainers.confidence  # noqa: F401
import crowdflow_dna.explainability.explainers.ablation  # noqa: F401
import crowdflow_dna.explainability.explainers.attention  # noqa: F401
import crowdflow_dna.explainability.reporting.markdown  # noqa: F401
import crowdflow_dna.explainability.reporting.json_report  # noqa: F401
import crowdflow_dna.explainability.reporting.csv_report  # noqa: F401
import crowdflow_dna.explainability.plotting.plots  # noqa: F401

class DummyModel(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr, batch, seq_lengths):
        return torch.randn(1, 2)
        
class DummyDataset:
    def __iter__(self):
        # yields sequence_id, data
        data = Data(
            x=torch.randn(10, 5),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.randn(2, 3),
            batch=torch.zeros(10, dtype=torch.long),
            seq_lengths=torch.tensor([10])
        )
        yield "seq_1", data

@pytest.fixture
def context():
    model = DummyModel()
    model.eval()
    runtime = InferenceRuntime()
    # Mocking the loaded model for tests
    from crowdflow_dna.inference.runtime import TorchScriptBackend
    backend = TorchScriptBackend()
    backend.model = model
    backend.device = "cpu"
    runtime.backend = backend
    runtime.backend_name = "TorchScript"
    
    dataset = DummyDataset()
    
    return ExplanationContext(
        model=model,
        runtime=runtime,
        dataset_provider=dataset,
        output_directory="test_outputs",
        configuration={"explainers": ["confidence", "ablation"]},
        logger=logging.getLogger(),
        device="cpu"
    )

def test_registry():
    metadata = ExplainerRegistry.get_metadata("confidence")
    assert metadata.category == ExplainerCategory.CONFIDENCE_BASED
    
def test_cache(context):
    session = ExplanationSession(
        session_id="test_1",
        crowddna_version="1.0",
        schema_version="1.0",
        deployment_model_hash="h1",
        deployment_artifact_hash="h2",
        dataset_identifier="d1",
        enabled_explainers=("confidence",),
        configuration_hash="c1",
        execution_timestamp="2026-01-01",
        execution_duration=1.0,
        output_directory="test_outputs",
        cache_status="enabled"
    )
    
    graph = ExplanationGraph(
        session=session,
        explainer_name="confidence",
        sequence_id="seq_1",
        node_importance=None,
        edge_importance=None,
        temporal_importance=None,
        confidence_analysis={"entropy": 0.5, "margin": 0.1},
        prediction_metadata={}
    )
    
    cache = CacheManager(cache_dir="test_cache")
    cache.set(graph)
    
    cached_graph = cache.get(session, "confidence", "seq_1")
    assert cached_graph is not None
    assert cached_graph.confidence_analysis["entropy"] == 0.5
    
    # Clean up
    if os.path.exists("test_cache"):
        shutil.rmtree("test_cache")

def test_pipeline(context):
    if os.path.exists("test_outputs"):
        shutil.rmtree("test_outputs")
        
    pipeline = ExplanationPipeline(output_dir="test_outputs")
    pipeline.run(context)
    
    assert os.path.exists("test_outputs/explanation_report.md")
    assert os.path.exists("test_outputs/explanation_report.json")
    assert os.path.exists("test_outputs/explanation_report.csv")
    assert os.path.exists("test_outputs/plots")
    
    shutil.rmtree("test_outputs")

def test_registry_duplicate():
    from crowdflow_dna.explainability.registry import ExplainerMetadata, ExplainerCategory
    metadata = ExplainerMetadata(
        name="duplicate_test",
        category=ExplainerCategory.CONFIDENCE_BASED,
        supports_batch=False,
        requires_gradients=False,
        requires_hooks=False,
        deterministic=True,
        runtime_cost="LOW",
        description="test"
    )
    
    @ExplainerRegistry.register(metadata)
    class TestExplainer1:
        pass
    
    with pytest.raises(ValueError, match="already registered"):
        @ExplainerRegistry.register(metadata)
        class TestExplainer2:
            pass

def test_cache_corruption(context):
    session = ExplanationSession(
        session_id="test_corrupt",
        crowddna_version="1.0",
        schema_version="1.0",
        deployment_model_hash="h1",
        deployment_artifact_hash="h2",
        dataset_identifier="d1",
        enabled_explainers=("confidence",),
        configuration_hash="c1",
        execution_timestamp="2026-01-01",
        execution_duration=1.0,
        output_directory="test_outputs",
        cache_status="enabled"
    )
    cache = CacheManager(cache_dir="test_cache_corrupt")
    key = cache._compute_hash(session, "confidence", "seq_1")
    path = os.path.join("test_cache_corrupt", f"{key}.pkl")
    
    os.makedirs("test_cache_corrupt", exist_ok=True)
    with open(path, "w") as f:
        f.write("corrupted data")
        
    result = cache.get(session, "confidence", "seq_1")
    assert result is None
    assert not os.path.exists(path) # File should be deleted
    
    if os.path.exists("test_cache_corrupt"):
        shutil.rmtree("test_cache_corrupt")
