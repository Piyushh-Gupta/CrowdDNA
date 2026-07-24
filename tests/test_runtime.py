import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from crowdflow_dna.inference.runtime import (
    InferenceExecutionError,
    InferenceResult,
    InferenceRuntime,
    ModelNotFoundError,
    UnsupportedModelFormatError,
)


@pytest.fixture
def mock_dummy_inputs():
    x = torch.randn(10, 3)
    edge_index = torch.randint(0, 10, (2, 15))
    edge_attr = torch.randn(15, 4)
    batch = torch.zeros(10, dtype=torch.long)
    seq_lengths = torch.tensor([10])
    return x, edge_index, edge_attr, batch, seq_lengths


@pytest.fixture
def mock_batch_inputs():
    x = torch.randn(20, 3)
    edge_index = torch.randint(0, 20, (2, 30))
    edge_attr = torch.randn(30, 4)
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    seq_lengths = torch.tensor([10, 10])
    return x, edge_index, edge_attr, batch, seq_lengths


@pytest.fixture
def dummy_torchscript_file(tmp_path):
    class DummyTSModel(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr, batch, seq_lengths):
            # return logits for 1 item (batch_size=1, num_classes=2)
            # wait, if seq_lengths is 1, size is 1. If 2, size is 2.
            batch_size = seq_lengths.size(0)
            return torch.zeros((batch_size, 2))
    
    model = torch.jit.script(DummyTSModel())
    path = tmp_path / "model.pt"
    torch.jit.save(model, path)
    return path


@pytest.fixture
def dummy_onnx_file(tmp_path):
    path = tmp_path / "model.onnx"
    # we just touch the file since we will mock onnxruntime anyway
    path.touch()
    return path


def test_runtime_load_torchscript(dummy_torchscript_file, mock_dummy_inputs):
    runtime = InferenceRuntime()
    runtime.load_model(dummy_torchscript_file, version="1.0")
    
    assert runtime.model_format == "TorchScript"
    assert runtime.backend_name == "TorchScript"
    assert runtime.model_version == "1.0"
    
    x, edge_index, edge_attr, batch, seq_lengths = mock_dummy_inputs
    result = runtime.predict(x, edge_index, edge_attr, batch, seq_lengths)
    
    assert isinstance(result, InferenceResult)
    assert result.backend == "TorchScript"
    assert result.model_format == "TorchScript"
    assert result.model_version == "1.0"
    assert isinstance(result.probabilities, np.ndarray)
    assert result.inference_time_ms >= 0


def test_runtime_predict_batch_torchscript(dummy_torchscript_file, mock_batch_inputs):
    runtime = InferenceRuntime()
    runtime.load_model(dummy_torchscript_file)
    
    x, edge_index, edge_attr, batch, seq_lengths = mock_batch_inputs
    results = runtime.predict_batch(x, edge_index, edge_attr, batch, seq_lengths)
    
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, InferenceResult)
        assert isinstance(result.probabilities, np.ndarray)


def test_runtime_load_onnx(dummy_onnx_file, mock_dummy_inputs):
    runtime = InferenceRuntime()
    
    # Mock onnxruntime
    mock_ort = MagicMock()
    mock_session = MagicMock()
    # return logits array for batch_size=1
    mock_session.run.return_value = [np.array([[0.5, -0.5]])]
    mock_ort.InferenceSession.return_value = mock_session
    
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        runtime.load_model(dummy_onnx_file)
        assert runtime.model_format == "ONNX"
        assert runtime.backend_name == "ONNX Runtime"
        
        x, edge_index, edge_attr, batch, seq_lengths = mock_dummy_inputs
        result = runtime.predict(x, edge_index, edge_attr, batch, seq_lengths)
        
        assert isinstance(result, InferenceResult)
        assert result.backend == "ONNX Runtime"
        assert isinstance(result.probabilities, np.ndarray)
        mock_session.run.assert_called_once()


def test_runtime_model_not_found():
    runtime = InferenceRuntime()
    with pytest.raises(ModelNotFoundError):
        runtime.load_model(Path("non_existent_model.pt"))


def test_runtime_unsupported_format(tmp_path):
    path = tmp_path / "model.h5"
    path.touch()
    
    runtime = InferenceRuntime()
    with pytest.raises(UnsupportedModelFormatError):
        runtime.load_model(path)


def test_runtime_predict_without_load(mock_dummy_inputs):
    runtime = InferenceRuntime()
    x, edge_index, edge_attr, batch, seq_lengths = mock_dummy_inputs
    with pytest.raises(InferenceExecutionError, match="No model loaded"):
        runtime.predict(x, edge_index, edge_attr, batch, seq_lengths)


def test_runtime_torchscript_execution_error(dummy_torchscript_file, mock_dummy_inputs):
    runtime = InferenceRuntime()
    runtime.load_model(dummy_torchscript_file)
    
    # Intentionally corrupt inputs to trigger TorchScript failure
    # seq_lengths expects 1D, pass empty tuple/list/wrong shape
    with pytest.raises(InferenceExecutionError, match="TorchScript inference failed"):
        runtime.predict(None, None, None, None, None) # type: ignore
