import json
from unittest.mock import patch

import pytest
import torch

from crowdflow_dna.model.crowddna_model import CrowdDNAModelConfig
from crowdflow_dna.model.gat_model import GATConfig
from crowdflow_dna.model.temporal_encoder import TemporalConfig
from training.export_model import ExportResult, ModelExporter


@pytest.fixture
def exporter_config():
    return {
        "input_dim": 3,
        "gnn_hidden_dim": 8,
        "classes": ["A", "B", "C"],
        "num_gnn_layers": 2,
        "gnn_heads": 2,
        "dropout": 0.0,
        "gru_hidden_dim": 8,
        "gru_num_layers": 1,
        "gru_dropout": 0.0,
        "gru_bidirectional": False
    }


@pytest.fixture
def mock_checkpoint(tmp_path):
    # Create a real checkpoint with properly initialized weights
    ckpt_path = tmp_path / "best.pt"
    
    # We must instantiate a real deployment model to get valid weights for loading
    config = CrowdDNAModelConfig(
        gat_config=GATConfig(
            in_channels=3,
            hidden_channels=8,
            out_channels=3,
            num_layers=2,
            heads=2,
            dropout=0.0
        ),
        temporal_config=TemporalConfig(
            input_dim=8,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        ),
        num_classes=3
    )
    from crowdflow_dna.model.deployment_model import CrowdDNADeploymentModel
    model = CrowdDNADeploymentModel(config)
    
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    return str(ckpt_path)


def test_exporter_execution_success(exporter_config, mock_checkpoint, tmp_path):
    exporter = ModelExporter(
        config_dict=exporter_config,
        checkpoint_path=mock_checkpoint,
        export_dir=str(tmp_path / "exports"),
        opset_version=17,
        tolerance=1e-3
    )
    
    # Ensure onnxruntime validation happens if installed, or falls back safely
    result = exporter.export()
    
    assert isinstance(result, ExportResult)
    assert result.validation_success is True
    assert result.torchscript_path.exists()
    if result.onnx_path:
        assert result.onnx_path.exists()
    assert result.metadata_path.exists()
    
    with open(result.metadata_path, "r") as f:
        meta = json.load(f)
        
    assert meta["validation_status"] == "SUCCESS"
    assert meta["export_method"] == "torch.jit.script"
    assert meta["opset_version"] == 17


def test_exporter_missing_onnxruntime(exporter_config, mock_checkpoint, tmp_path):
    exporter = ModelExporter(
        config_dict=exporter_config,
        checkpoint_path=mock_checkpoint,
        export_dir=str(tmp_path / "exports_no_ort")
    )
    
    import sys
    with patch.dict(sys.modules, {"onnxruntime": None}):
        result = exporter.export()
        assert result.validation_success is True
        if result.onnx_path:
            assert result.onnx_path.exists()
