import pytest
import torch
from torch_geometric.data import Batch, Data

from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig
from crowdflow_dna.model.gat_model import GATConfig
from crowdflow_dna.model.temporal_encoder import TemporalConfig


@pytest.fixture
def valid_gat_config():
    return GATConfig(
        in_channels=5,
        hidden_channels=64,
        out_channels=3,
        num_layers=2,
        heads=4,
        dropout=0.2,
    )


@pytest.fixture
def valid_temporal_config():
    return TemporalConfig(
        input_dim=64,
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )


@pytest.fixture
def valid_model_config(valid_gat_config, valid_temporal_config):
    return CrowdDNAModelConfig(
        gat_config=valid_gat_config,
        temporal_config=valid_temporal_config,
        num_classes=3,
    )


@pytest.fixture
def dummy_sequence_batch():
    """Creates a list of 2 sequences, one with 3 frames and one with 2 frames."""
    seq1 = [
        Data(x=torch.randn((4, 5)), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn((2, 4))),
        Data(x=torch.randn((4, 5)), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn((2, 4))),
        Data(x=torch.randn((4, 5)), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn((2, 4))),
    ]
    seq2 = [
        Data(x=torch.randn((3, 5)), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn((2, 4))),
        Data(x=torch.randn((3, 5)), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), edge_attr=torch.randn((2, 4))),
    ]
    return [seq1, seq2]


def test_model_construction(valid_model_config):
    """Test successful initialization of the model."""
    model = CrowdDNAModel(valid_model_config)
    assert model.gat is not None
    assert model.temporal_encoder is not None
    assert model.classifier.out_features == 3
    # Unidirectional GRU means temporal_out_dim = hidden_dim = 32
    assert model.classifier.in_features == 32


def test_dimension_mismatch_detection(valid_gat_config, valid_temporal_config):
    """Test validation failure when GAT hidden_channels != Temporal input_dim."""
    # Create mismatch
    mismatched_temporal = TemporalConfig(
        input_dim=128,  # Does not match GAT's 64
        hidden_dim=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )
    with pytest.raises(ValueError, match="Dimension mismatch: GAT hidden_channels"):
        config = CrowdDNAModelConfig(
            gat_config=valid_gat_config,
            temporal_config=mismatched_temporal,
            num_classes=3,
        )
        CrowdDNAModel(config)


def test_invalid_num_classes(valid_gat_config, valid_temporal_config):
    """Test validation failure when num_classes < 1."""
    with pytest.raises(ValueError, match="num_classes must be >= 1"):
        config = CrowdDNAModelConfig(
            gat_config=valid_gat_config,
            temporal_config=valid_temporal_config,
            num_classes=0,
        )
        CrowdDNAModel(config)


def test_forward_pass_list_of_lists(valid_model_config, dummy_sequence_batch):
    """Test forward pass using a list of lists of Data objects."""
    model = CrowdDNAModel(valid_model_config)
    
    out = model(dummy_sequence_batch)
    assert out.shape == (2, 3)  # (batch_size=2 sequences, num_classes=3)


def test_forward_pass_list_of_batches(valid_model_config, dummy_sequence_batch):
    """Test forward pass using a list of Batch objects (one Batch per sequence)."""
    model = CrowdDNAModel(valid_model_config)
    
    # Convert inner lists to PyG Batch objects
    batch_seq1 = Batch.from_data_list(dummy_sequence_batch[0])
    batch_seq2 = Batch.from_data_list(dummy_sequence_batch[1])
    
    out = model([batch_seq1, batch_seq2])
    assert out.shape == (2, 3)


def test_forward_empty_input(valid_model_config):
    """Test that model rejects empty input gracefully."""
    model = CrowdDNAModel(valid_model_config)
    
    with pytest.raises(ValueError, match="received an empty batch of sequences"):
        model([])


def test_forward_empty_sequence(valid_model_config):
    """Test that model rejects empty inner sequences."""
    model = CrowdDNAModel(valid_model_config)
    
    with pytest.raises(ValueError, match="Encountered an empty sequence in the batch"):
        model([[], []])


def test_gradient_propagation(valid_model_config, dummy_sequence_batch):
    """Test that gradients propagate through both TemporalEncoder and GAT."""
    model = CrowdDNAModel(valid_model_config)
    model.train()
    
    out = model(dummy_sequence_batch)
    loss = out.sum()
    loss.backward()
    
    # Check if classifier has gradients
    assert model.classifier.weight.grad is not None
    # Check if TemporalEncoder has gradients
    assert model.temporal_encoder.gru.weight_ih_l0.grad is not None
    # Check if GAT has gradients
    assert model.gat.convs[0].lin.weight.grad is not None


def test_deterministic_evaluation_mode(valid_model_config, dummy_sequence_batch):
    """Test that eval mode is deterministic (dropout is disabled)."""
    model = CrowdDNAModel(valid_model_config)
    model.eval()
    
    out1 = model(dummy_sequence_batch)
    out2 = model(dummy_sequence_batch)
    
    assert torch.allclose(out1, out2)


def test_train_eval_behaviour(valid_model_config, dummy_sequence_batch):
    """Test that train mode produces different outputs due to dropout."""
    model = CrowdDNAModel(valid_model_config)
    model.train()
    
    out1 = model(dummy_sequence_batch)
    out2 = model(dummy_sequence_batch)
    
    # Probabilistic test, dropout=0.2 means it's extremely unlikely to be identical
    assert not torch.allclose(out1, out2)


def test_config_from_dict():
    """Test loading the composed configuration from a dictionary."""
    config_dict = {
        "input_dim": 5,           # GAT input
        "gnn_hidden_dim": 128,    # GAT hidden and Temporal input
        "gru_hidden_dim": 64,     # Temporal hidden
        "gru_num_layers": 3,
        "gru_dropout": 0.5,
        "gru_bidirectional": True,
        "num_gnn_layers": 3,
        "gnn_heads": 4,
        "classes": ["A", "B"],    # 2 classes
    }
    
    config = CrowdDNAModelConfig.from_dict(config_dict)
    
    assert config.num_classes == 2
    assert config.gat_config.hidden_channels == 128
    assert config.temporal_config.input_dim == 128
    assert config.temporal_config.hidden_dim == 64
    assert config.temporal_config.bidirectional is True
    
    # Should not raise exception
    config.validate()
