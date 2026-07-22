import torch
from torch_geometric.data import Data, Batch
from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig

def test_gat_per_sequence_equivalence():
    """
    Validates that the new per-sequence GAT implementation is mathematically
    equivalent to the old giant-batch implementation.
    """
    torch.manual_seed(42)
    
    # 1. Setup small model
    config_dict = {
        "input_dim": 3,
        "gnn_hidden_dim": 8,
        "num_gnn_layers": 2,
        "gnn_heads": 2,
        "dropout": 0.0,  # Disable dropout for deterministic equivalence
        "gru_hidden_dim": 16,
        "gru_num_layers": 1,
        "gru_bidirectional": False,
        "gru_dropout": 0.0,
        "classes": ["A", "B"]
    }
    config = CrowdDNAModelConfig.from_dict(config_dict)
    
    # We need two identical models to isolate gradient accumulation
    model_new = CrowdDNAModel(config)
    model_old = CrowdDNAModel(config)
    
    # Copy weights exactly
    model_old.load_state_dict(model_new.state_dict())
    
    model_new.eval()
    model_old.eval()
    
    # 2. Generate synthetic data
    # 2 sequences, 3 frames each
    sequences = []
    for _ in range(2):
        seq = []
        for _ in range(3):
            # 5 nodes per frame
            x = torch.randn((5, 3))
            # fully connected edges (for simplicity)
            edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 3],
                [1, 2, 0, 2, 1, 4]
            ])
            edge_attr = torch.randn((6, 4))
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        sequences.append(seq)
        
    labels = torch.tensor([0, 1])

    # 3. Old implementation (Giant Batch)
    def old_forward(model, sequences):
        flat_graphs = []
        seq_lengths = []
        for seq in sequences:
            flat_graphs.extend(seq)
            seq_lengths.append(len(seq))
            
        giant_batch = Batch.from_data_list(flat_graphs)
        graph_embeddings = model.gat.extract_features(giant_batch)
        
        batch_size = len(sequences)
        max_seq_len = max(seq_lengths)
        embedding_dim = graph_embeddings.size(-1)
        
        padded_sequences = torch.zeros(
            (batch_size, max_seq_len, embedding_dim), 
            device=graph_embeddings.device,
            dtype=graph_embeddings.dtype
        )
        
        start_idx = 0
        for b_idx, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len
            padded_sequences[b_idx, :seq_len, :] = graph_embeddings[start_idx:end_idx]
            start_idx = end_idx
            
        temporal_rep = model.temporal_encoder(padded_sequences)
        logits = model.classifier(temporal_rep)
        return logits

    model_old.train()
    model_new.train()
    
    # Run old
    logits_old = old_forward(model_old, sequences)
    loss_old = torch.nn.functional.cross_entropy(logits_old, labels)
    loss_old.backward()
    
    # Run new
    logits_new = model_new(sequences)
    loss_new = torch.nn.functional.cross_entropy(logits_new, labels)
    loss_new.backward()
    
    # 4. Assertions
    # Logits should match exactly
    assert logits_old.shape == logits_new.shape, "Output shapes differ"
    assert torch.allclose(logits_old, logits_new, atol=1e-5), "Logits are not equivalent"
    
    # Gradients should match exactly for GAT parameters
    for (name_old, p_old), (name_new, p_new) in zip(model_old.named_parameters(), model_new.named_parameters()):
        assert name_old == name_new
        if p_old.grad is not None and p_new.grad is not None:
            assert torch.allclose(p_old.grad, p_new.grad, atol=1e-5), f"Gradient mismatch in {name_old}"
