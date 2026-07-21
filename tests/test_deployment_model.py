import pytest
import torch
from torch_geometric.data import Batch, Data

from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig
from crowdflow_dna.model.deployment_model import CrowdDNADeploymentModel
from crowdflow_dna.model.gat_model import GATConfig
from crowdflow_dna.model.temporal_encoder import TemporalConfig


@pytest.fixture
def model_config():
    gat_config = GATConfig(
        in_channels=3,
        hidden_channels=8,
        out_channels=8,
        num_layers=2,
        heads=2,
        dropout=0.0
    )
    temporal_config = TemporalConfig(
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        bidirectional=False
    )
    return CrowdDNAModelConfig(
        gat_config=gat_config,
        temporal_config=temporal_config,
        num_classes=3
    )


@pytest.fixture
def sample_data():
    """Generates synthetic PyG Data graphs and unrolls them for deployment testing."""
    torch.manual_seed(42)
    
    # 2 trajectories, each with 3 frames
    seqs = []
    flat_graphs = []
    
    for i in range(2):
        traj = []
        for j in range(3):
            # Synthetic graph
            x = torch.rand(5, 3) # 5 nodes, 3 features
            
            # Synthetic dense edge index
            edge_index = torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 0]
            ], dtype=torch.long)
            
            edge_attr = torch.rand(5, 4) # 5 edges, 4 features
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            traj.append(data)
            flat_graphs.append(data)
            
        seqs.append(traj)
        
    # Unroll into flat tensors for deployment
    giant_batch = Batch.from_data_list(flat_graphs)
    seq_lengths = torch.tensor([3, 3], dtype=torch.long)
    
    return seqs, giant_batch, seq_lengths


def test_deployment_numerical_equivalence(model_config, sample_data):
    """Verifies that the deployment model produces exactly the same logits as the training model."""
    seqs, giant_batch, seq_lengths = sample_data
    
    # Initialize training model
    torch.manual_seed(100)
    train_model = CrowdDNAModel(model_config)
    train_model.eval()
    
    # Initialize deployment model and inject weights
    deploy_model = CrowdDNADeploymentModel(model_config)
    deploy_model.load_from_training_model(train_model)
    deploy_model.eval()
    
    with torch.no_grad():
        # Forward pass on training model (expects list of lists of Data)
        train_logits = train_model(seqs)
        
        # Forward pass on deployment model (expects flat tensors)
        deploy_logits = deploy_model(
            x=giant_batch.x,
            edge_index=giant_batch.edge_index,
            edge_attr=giant_batch.edge_attr,
            batch=giant_batch.batch,
            seq_lengths=seq_lengths
        )
        
    assert torch.allclose(train_logits, deploy_logits, atol=1e-6)


def test_deployment_torchscript_scriptable(model_config, sample_data):
    """Verifies that the deployment model can be compiled by TorchScript via torch.jit.script."""
    _, giant_batch, seq_lengths = sample_data
    
    deploy_model = CrowdDNADeploymentModel(model_config)
    deploy_model.eval()
    
    # Script the model
    try:
        scripted_model = torch.jit.script(deploy_model)
    except Exception as e:
        pytest.fail(f"torch.jit.script failed on CrowdDNADeploymentModel: {e}")
        
    with torch.no_grad():
        orig_out = deploy_model(
            x=giant_batch.x,
            edge_index=giant_batch.edge_index,
            edge_attr=giant_batch.edge_attr,
            batch=giant_batch.batch,
            seq_lengths=seq_lengths
        )
        
        scripted_out = scripted_model(
            giant_batch.x,
            giant_batch.edge_index,
            giant_batch.edge_attr,
            giant_batch.batch,
            seq_lengths
        )
        
    assert torch.allclose(orig_out, scripted_out, atol=1e-6)
