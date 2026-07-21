import logging
from dataclasses import FrozenInstanceError

import pytest
import torch

from crowdflow_dna.model.temporal_encoder import TemporalConfig, TemporalEncoder


@pytest.fixture
def valid_config():
    return TemporalConfig(
        input_dim=64,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )


@pytest.fixture
def valid_bidirectional_config():
    return TemporalConfig(
        input_dim=64,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
    )


def test_model_construction(valid_config):
    """Test successful initialization of the model."""
    model = TemporalEncoder(valid_config)
    assert model.gru.input_size == 64
    assert model.gru.hidden_size == 32
    assert model.gru.num_layers == 2
    assert model.gru.dropout == 0.2
    assert model.gru.bidirectional is False
    assert model.gru.batch_first is True


def test_invalid_configuration():
    """Test validation of configuration parameters."""
    with pytest.raises(ValueError, match="input_dim"):
        config = TemporalConfig(0, 32, 2, 0.2, False)
        TemporalEncoder(config)

    with pytest.raises(ValueError, match="hidden_dim"):
        config = TemporalConfig(64, -1, 2, 0.2, False)
        TemporalEncoder(config)

    with pytest.raises(ValueError, match="num_layers"):
        config = TemporalConfig(64, 32, 0, 0.2, False)
        TemporalEncoder(config)

    with pytest.raises(ValueError, match="dropout"):
        config = TemporalConfig(64, 32, 2, 1.2, False)
        TemporalEncoder(config)


def test_single_layer_dropout_warning(caplog):
    """Test that setting dropout > 0 with num_layers=1 emits a warning."""
    with caplog.at_level(logging.WARNING):
        config = TemporalConfig(64, 32, 1, 0.5, False)
        TemporalEncoder(config)
    
    assert "PyTorch GRU ignores dropout for a single layer" in caplog.text


def test_frozen_config_cannot_be_mutated(valid_config):
    """Test that the configuration cannot be mutated after creation."""
    with pytest.raises(FrozenInstanceError):
        valid_config.dropout = 0.5


def test_forward_pass_and_shape(valid_config):
    """Test the forward pass and verify output shape."""
    model = TemporalEncoder(valid_config)
    
    batch_size = 4
    seq_len = 10
    x = torch.randn((batch_size, seq_len, valid_config.input_dim))
    
    out = model(x)
    assert out.shape == (batch_size, valid_config.hidden_dim)


def test_bidirectional_mode(valid_bidirectional_config):
    """Test the forward pass in bidirectional mode."""
    model = TemporalEncoder(valid_bidirectional_config)
    
    batch_size = 4
    seq_len = 10
    x = torch.randn((batch_size, seq_len, valid_bidirectional_config.input_dim))
    
    out = model(x)
    # Output dim should be hidden_dim * 2
    assert out.shape == (batch_size, valid_bidirectional_config.hidden_dim * 2)


def test_gradient_propagation(valid_config):
    """Test that gradients propagate through the model parameters."""
    model = TemporalEncoder(valid_config)
    model.train()
    
    x = torch.randn((4, 10, valid_config.input_dim))
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # Check if GRU weights have gradients
    assert model.gru.weight_ih_l0.grad is not None
    assert model.gru.weight_hh_l0.grad is not None


def test_deterministic_evaluation_mode(valid_config):
    """Test that eval mode is deterministic (dropout is disabled)."""
    # Ensure config has enough dropout to cause a difference in train mode
    config = TemporalConfig(
        input_dim=64,
        hidden_dim=32,
        num_layers=2,
        dropout=0.8,
        bidirectional=False,
    )
    model = TemporalEncoder(config)
    model.eval()
    
    x = torch.randn((4, 10, config.input_dim))
    
    out1 = model(x)
    out2 = model(x)
    
    assert torch.allclose(out1, out2)


def test_train_eval_behaviour():
    """Test that train mode produces different outputs due to dropout."""
    config = TemporalConfig(
        input_dim=64,
        hidden_dim=32,
        num_layers=2,
        dropout=0.8,
        bidirectional=False,
    )
    model = TemporalEncoder(config)
    model.train()
    
    x = torch.randn((4, 10, config.input_dim))
    
    out1 = model(x)
    out2 = model(x)
    
    assert not torch.allclose(out1, out2)


def test_variable_batch_sizes(valid_config):
    """Test that the model handles different batch sizes."""
    model = TemporalEncoder(valid_config)
    
    x_small = torch.randn((1, 10, valid_config.input_dim))
    out_small = model(x_small)
    assert out_small.shape == (1, valid_config.hidden_dim)
    
    x_large = torch.randn((16, 10, valid_config.input_dim))
    out_large = model(x_large)
    assert out_large.shape == (16, valid_config.hidden_dim)


def test_variable_sequence_lengths(valid_config):
    """Test that the model handles different sequence lengths."""
    model = TemporalEncoder(valid_config)
    
    x_short = torch.randn((4, 1, valid_config.input_dim))
    out_short = model(x_short)
    assert out_short.shape == (4, valid_config.hidden_dim)
    
    x_long = torch.randn((4, 100, valid_config.input_dim))
    out_long = model(x_long)
    assert out_long.shape == (4, valid_config.hidden_dim)


def test_config_from_dict():
    """Test loading configuration from a dictionary."""
    config_dict = {
        "gnn_hidden_dim": 128,
        "gru_hidden_dim": 64,
        "gru_num_layers": 3,
        "gru_dropout": 0.5,
        "gru_bidirectional": True,
    }
    
    config = TemporalConfig.from_dict(config_dict)
    
    assert config.input_dim == 128
    assert config.hidden_dim == 64
    assert config.num_layers == 3
    assert config.dropout == 0.5
    assert config.bidirectional is True
