import pytest
import torch
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data
from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig

def test_gradient_checkpointing_active():
    """
    Verifies that gradient checkpointing is actively recomputing the GAT forward pass
    during the backward propagation phase, preventing intermediate tensor accumulation.
    """
    # 1. Setup small model
    config = CrowdDNAModelConfig.from_dict({
        "input_dim": 3,
        "gnn_hidden_dim": 8,
        "num_gnn_layers": 1,
        "gnn_heads": 1,
        "dropout": 0.0,
        "gru_hidden_dim": 16,
        "gru_num_layers": 1,
        "gru_bidirectional": False,
        "gru_dropout": 0.0,
        "classes": ["A", "B"]
    })
    
    model = CrowdDNAModel(config)
    model.train()
    
    # 2. Generate 2 synthetic sequences
    sequences = []
    for _ in range(2):
        seq = []
        for _ in range(2): # 2 frames
            x = torch.randn((5, 3))
            edge_index = torch.tensor([[0, 1], [1, 0]])
            edge_attr = torch.randn((2, 4))
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        sequences.append(seq)
        
    labels = torch.tensor([0, 1])
    
    # We will spy on the extract_features method
    original_extract = model.gat.extract_features
    
    call_count = 0
    def spy_extract(batch):
        nonlocal call_count
        call_count += 1
        return original_extract(batch)
    
    # 3. Patch the method and execute forward and backward passes
    with patch.object(model.gat, 'extract_features', side_effect=spy_extract) as mock_extract:
        # Forward pass
        logits = model(sequences)
        
        # In a batch of 2 sequences, extract_features should be called 2 times
        # during the forward pass.
        assert call_count == 2, "Forward pass should call extract_features exactly twice (once per sequence)."
        
        # Backward pass
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        
        # If gradient checkpointing is active, extract_features MUST be called again 
        # during the backward pass to recompute the discarded intermediate activations.
        # It should add exactly 2 more calls.
        assert call_count == 4, (
            "Gradient checkpointing failed! The GAT forward pass was not recomputed "
            "during the backward pass. Ensure seq_batch.x.requires_grad=True is set."
        )
