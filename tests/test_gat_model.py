import pytest
import torch
from torch_geometric.data import Data, Batch

from crowdflow_dna.model.gat_model import CrowdDNAGAT, GATConfig


@pytest.fixture
def valid_config():
    return GATConfig(
        in_channels=5,
        hidden_channels=64,
        out_channels=3,
        num_layers=2,
        heads=4,
        dropout=0.2,
    )


def test_model_construction(valid_config):
    """Test successful initialization of the model."""
    model = CrowdDNAGAT(valid_config)
    assert len(model.convs) == 2
    assert model.classifier.out_features == 3
    assert model.classifier.in_features == 64


def test_invalid_configuration():
    """Test validation of configuration parameters."""
    with pytest.raises(ValueError, match="in_channels"):
        config = GATConfig(0, 64, 3, 2, 4, 0.2)
        CrowdDNAGAT(config)

    with pytest.raises(ValueError, match="hidden_channels"):
        config = GATConfig(5, -1, 3, 2, 4, 0.2)
        CrowdDNAGAT(config)

    with pytest.raises(ValueError, match="out_channels"):
        config = GATConfig(5, 64, 0, 2, 4, 0.2)
        CrowdDNAGAT(config)

    with pytest.raises(ValueError, match="num_layers"):
        config = GATConfig(5, 64, 3, 0, 4, 0.2)
        CrowdDNAGAT(config)

    with pytest.raises(ValueError, match="heads"):
        config = GATConfig(5, 64, 3, 2, 0, 0.2)
        CrowdDNAGAT(config)

    with pytest.raises(ValueError, match="dropout"):
        config = GATConfig(5, 64, 3, 2, 4, 1.2)
        CrowdDNAGAT(config)


def test_forward_pass_and_shape(valid_config):
    """Test the forward pass on a single graph and verify output shape."""
    model = CrowdDNAGAT(valid_config)
    # 4 nodes, 5 features
    x = torch.randn((4, 5))
    # 2 edges (bidirectional)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    out = model(data)
    # Output should be (1 batch, 3 classes)
    assert out.shape == (1, 3)


def test_batched_graphs(valid_config):
    """Test forward pass on batched graphs."""
    model = CrowdDNAGAT(valid_config)
    
    # Graph 1: 3 nodes
    x1 = torch.randn((3, 5))
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)
    
    # Graph 2: 2 nodes
    x2 = torch.randn((2, 5))
    edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)
    
    batch = Batch.from_data_list([data1, data2])
    out = model(batch)
    
    # Output should be (2 graphs in batch, 3 classes)
    assert out.shape == (2, 3)


def test_empty_graph_handling(valid_config):
    """Test that the model gracefully handles a graph with zero nodes."""
    model = CrowdDNAGAT(valid_config)
    x = torch.empty((0, 5))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    out = model(data)
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.zeros((1, 3)))


def test_gradient_propagation(valid_config):
    """Test that gradients propagate through the model parameters."""
    model = CrowdDNAGAT(valid_config)
    model.train()
    
    x = torch.randn((4, 5))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    out = model(data)
    loss = out.sum()
    loss.backward()
    
    # Check if classifier weight has gradients
    assert model.classifier.weight.grad is not None
    # Check if first conv layer has gradients
    assert model.convs[0].lin.weight.grad is not None


def test_deterministic_evaluation_mode(valid_config):
    """Test that eval mode is deterministic (dropout is disabled)."""
    model = CrowdDNAGAT(valid_config)
    model.eval()
    
    x = torch.randn((4, 5))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    out1 = model(data)
    out2 = model(data)
    
    assert torch.allclose(out1, out2)


def test_train_eval_behaviour(valid_config):
    """Test that train mode produces different outputs due to dropout."""
    model = CrowdDNAGAT(valid_config)
    # Ensure dropout is high enough to likely cause a difference
    model.config.dropout = 0.5
    
    # For GATConv, dropout is applied during training internally.
    # We must ensure we test in training mode.
    model.train()
    
    x = torch.randn((4, 5))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    out1 = model(data)
    out2 = model(data)
    
    # Since dropout is active, outputs should differ.
    # Note: this has a very small chance of failing if dropout randomly drops the same nodes,
    # but with multiple features/heads, it's virtually impossible.
    assert not torch.allclose(out1, out2)

def test_config_from_dict():
    """Test loading configuration from a dictionary."""
    config_dict = {
        "input_dim": 10,
        "gnn_hidden_dim": 128,
        "classes": ["A", "B", "C", "D"],
        "num_gnn_layers": 3,
        "gnn_heads": 8,
        "dropout": 0.5,
    }
    
    config = GATConfig.from_dict(config_dict)
    
    assert config.in_channels == 10
    assert config.hidden_channels == 128
    assert config.out_channels == 4
    assert config.num_layers == 3
    assert config.heads == 8
    assert config.dropout == 0.5
