import json
import yaml
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from crowdflow_dna.graph.sequence_dataset import SequenceSample
from training.train_model import (
    CheckpointManager,
    EarlyStopping,
    MetricsTracker,
    Trainer,
    TrainingBatch,
    sequence_collate_fn,
    set_random_seed,
)


@pytest.fixture
def mock_training_env(tmp_path):
    """Creates a temporary environment with configs and dataset for the Trainer."""
    # 1. Dataset Dir
    data_dir = tmp_path / "data" / "synthetic"
    data_dir.mkdir(parents=True)
    
    # Create 5 tiny sequences so train/val split works
    manifest = []
    for i in range(5):
        traj = {
            "sequence_id": f"seq_{i}",
            "risk_class": "Safe" if i % 2 == 0 else "Critical",
            "positions": [[[0.0, 0.0], [1.0, 1.0]], [[0.1, 0.1], [1.1, 1.1]]],
            "velocities": [[[0.1, 0.1], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]],
            "frame_labels": ["Safe", "Safe"]
        }
        fname = f"traj_{i}.json"
        with open(data_dir / fname, "w") as f:
            json.dump(traj, f)
        manifest.append({"file_path": fname, "num_timesteps": 2})
        
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    # 2. Config File
    config_dict = {
        "simulation": {
            "output_dir": str(data_dir)
        },
        "model": {
            "gnn_hidden_dim": 16,
            "gru_hidden_dim": 16,
            "gru_num_layers": 1,
            "gru_dropout": 0.0,
            "gru_bidirectional": False,
            "num_gnn_layers": 1,
            "classes": ["Safe", "Congesting", "Critical"],
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "num_epochs": 1,
            "patience": 2,
            "scheduler_patience": 1,
            "scheduler_factor": 0.5,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "random_seed": 42,
            "validation_split": 0.2, # 1 val sample, 4 train samples
        }
    }
    
    config_path = tmp_path / "default.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f)
        
    return str(config_path), str(tmp_path)


def test_set_random_seed():
    """Test deterministic seed."""
    set_random_seed(42)
    val1 = torch.rand(1).item()
    set_random_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2


def test_sequence_collate_fn():
    """Test collation of SequenceSample into TrainingBatch."""
    g1 = [Data(x=torch.randn(2, 5))]
    g2 = [Data(x=torch.randn(2, 5))]
    s1 = SequenceSample(sequence_id="s1", graphs=g1, label=torch.tensor(0))
    s2 = SequenceSample(sequence_id="s2", graphs=g2, label=torch.tensor(1))
    
    batch = sequence_collate_fn([s1, s2])
    
    assert isinstance(batch, TrainingBatch)
    assert len(batch.sequences) == 2
    assert batch.sequences[0] == g1
    assert batch.sequences[1] == g2
    assert batch.labels.shape == (2,)
    assert batch.labels.tolist() == [0, 1]


def test_metrics_tracker():
    """Test the MetricsTracker aggregates correctly."""
    tracker = MetricsTracker()
    
    # 2 correct out of 4, loss=1.0 per sample (total loss=4.0)
    tracker.update(loss=1.0, correct=2, total=4)
    # 3 correct out of 6, loss=0.5 per sample (total loss=3.0)
    tracker.update(loss=0.5, correct=3, total=6)
    
    metrics = tracker.compute()
    
    assert metrics["loss"] == 7.0 / 10.0
    assert metrics["accuracy"] == 5 / 10.0
    
    tracker.reset()
    metrics = tracker.compute()
    assert metrics["loss"] == 0.0


def test_early_stopping():
    """Test early stopping triggers after patience."""
    early_stopping = EarlyStopping(patience=2)
    
    early_stopping(1.0)
    assert not early_stopping.early_stop
    assert early_stopping.best_loss == 1.0
    assert early_stopping.counter == 0
    
    early_stopping(1.2) # worse
    assert not early_stopping.early_stop
    assert early_stopping.counter == 1
    
    early_stopping(1.3) # worse again
    assert early_stopping.early_stop
    assert early_stopping.counter == 2


def test_checkpoint_manager(tmp_path):
    """Test that CheckpointManager saves and loads states."""
    from torch.nn import Linear
    from torch.optim import SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    model = Linear(10, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer)
    
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(str(ckpt_dir))
    
    manager.save(model, optimizer, scheduler, epoch=5, best_val_metric=0.5, config={"test": 1}, is_best=True)
    
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    
    # Load into new instances
    new_model = Linear(10, 2)
    new_optimizer = SGD(new_model.parameters(), lr=0.5)
    
    epoch, best_val = manager.load(str(ckpt_dir / "best.pt"), new_model, new_optimizer)
    
    assert epoch == 5
    assert best_val == 0.5
    # Verify weights transferred
    assert torch.allclose(model.weight, new_model.weight)


def test_trainer_initialization(mock_training_env):
    """Test Trainer initializes components correctly."""
    config_path, _ = mock_training_env
    trainer = Trainer(config_path)
    
    # Verify dataset split
    assert len(trainer.train_dataset) == 4
    assert len(trainer.val_dataset) == 1
    
    # Verify dataloaders
    assert trainer.train_loader.batch_size == 2
    assert trainer.val_loader.batch_size == 2
    
    # Verify model/optim
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.early_stopping is not None


def test_trainer_one_step(mock_training_env):
    """Test one train and one val step."""
    config_path, _ = mock_training_env
    trainer = Trainer(config_path)
    
    train_metrics = trainer._train_epoch()
    assert "loss" in train_metrics
    assert "accuracy" in train_metrics
    
    val_metrics = trainer._validate_epoch()
    assert "loss" in val_metrics
    assert "accuracy" in val_metrics


def test_trainer_fit_and_resume(mock_training_env):
    """Test the full fit() loop and resuming from checkpoint."""
    config_path, tmp_path = mock_training_env
    
    # Set num_epochs to 2 for the test
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["num_epochs"] = 2
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)
        
    trainer = Trainer(config_path)
    history = trainer.fit()
    
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2
    
    ckpt_path = Path(tmp_path) / "checkpoints" / "latest.pt"
    assert ckpt_path.exists()
    
    # Resume
    trainer2 = Trainer(config_path, resume_checkpoint=str(ckpt_path))
    assert trainer2.start_epoch == 3 # 2 epochs were completed, so starts at 3
