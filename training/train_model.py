"""
CrowdFlow DNA — Training Pipeline
=================================
Module: training/train_model.py

Implements the end-to-end training infrastructure for the CrowdDNAModel.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset, SequenceSample
from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingBatch:
    """Represents a collated batch of trajectories ready for the model."""
    sequences: list[list[Data]]
    labels: torch.Tensor


def sequence_collate_fn(batch: list[SequenceSample]) -> TrainingBatch:
    """Custom collate function for SequenceSample dataclass.
    
    Extracts the lists of graphs and stacks the global labels.
    """
    sequences = [sample.graphs for sample in batch]
    labels = torch.stack([sample.label for sample in batch])
    return TrainingBatch(sequences=sequences, labels=labels)


@dataclass(frozen=True)
class TrainingHistory:
    """Structured container for training metrics history."""
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    best_val_loss: float
    best_epoch: int


class MetricsTracker:
    """Tracks and aggregates loss and accuracy over an epoch."""
    
    def __init__(self) -> None:
        self.reset()
        
    def update(self, loss: float, correct: int, total: int) -> None:
        self.total_loss += loss * total
        self.total_correct += correct
        self.total_samples += total
        
    def compute(self) -> dict[str, float]:
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": self.total_loss / self.total_samples,
            "accuracy": self.total_correct / self.total_samples,
        }
        
    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0


class EarlyStopping:
    """Monitors validation metric to halt training early if it plateaus."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        epoch: int,
        best_val_metric: float,
        config: dict[str, Any],
        is_best: bool = False,
    ) -> None:
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_val_metric": best_val_metric,
            "config": config,
        }
        
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            
    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> tuple[int, float]:
        """Loads state into the provided modules. Returns (start_epoch, best_val_metric)."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        
        if optimizer and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
            
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
            
        return state.get("epoch", 0), state.get("best_val_metric", float("inf"))


class Trainer:
    """Orchestrates the training pipeline."""
    
    def __init__(self, config_path: str, resume_checkpoint: str | None = None) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        train_cfg = self.config.get("training", {})
        self.seed = train_cfg.get("random_seed", 42)
        set_random_seed(self.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Dataset & DataLoaders
        self.dataset = SequenceGraphDataset(self.config["simulation"]["output_dir"])
        
        val_split = train_cfg.get("validation_split", 0.2)
        val_size = int(len(self.dataset) * val_split)
        train_size = len(self.dataset) - val_size
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=generator
        )
        
        batch_size = train_cfg.get("batch_size", 4)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=sequence_collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )
        
        # 2. Model Initialization
        model_cfg = CrowdDNAModelConfig.from_dict(self.config["model"])
        self.model = CrowdDNAModel(model_cfg).to(self.device)
        
        # 3. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=train_cfg.get("scheduler_factor", 0.5),
            patience=train_cfg.get("scheduler_patience", 5),
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # 4. Utilities
        self.early_stopping = EarlyStopping(patience=train_cfg.get("patience", 10))
        self.checkpoint_manager = CheckpointManager(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.metrics_tracker = MetricsTracker()
        
        self.start_epoch = 1
        
        # 5. Resume Support
        if resume_checkpoint:
            start_epoch, best_val = self.checkpoint_manager.load(
                resume_checkpoint, self.model, self.optimizer, self.scheduler
            )
            self.start_epoch = start_epoch + 1
            self.early_stopping.best_loss = best_val
            logger.info(f"Resumed from checkpoint {resume_checkpoint} at epoch {start_epoch}")

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.metrics_tracker.reset()
        
        for batch in self.train_loader:
            sequences_on_device = [
                [data.to(self.device) for data in seq] for seq in batch.sequences
            ]
            labels = batch.labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(sequences_on_device)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            self.metrics_tracker.update(loss.item(), correct, labels.size(0))
            
        return self.metrics_tracker.compute()
        
    def _validate_epoch(self) -> dict[str, float]:
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in self.val_loader:
                sequences_on_device = [
                    [data.to(self.device) for data in seq] for seq in batch.sequences
                ]
                labels = batch.labels.to(self.device)
                
                logits = self.model(sequences_on_device)
                loss = self.criterion(logits, labels)
                
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).sum().item()
                self.metrics_tracker.update(loss.item(), correct, labels.size(0))
                
        return self.metrics_tracker.compute()
        
    def fit(self) -> TrainingHistory:
        """Executes the complete training loop."""
        train_cfg = self.config.get("training", {})
        num_epochs = train_cfg.get("num_epochs", 50)
        
        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []
        
        best_val_loss = self.early_stopping.best_loss
        best_epoch = self.start_epoch - 1
        
        logger.info("Starting training pipeline...")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()
            
            if self.scheduler:
                self.scheduler.step(val_metrics["loss"])
                
            train_loss_hist.append(train_metrics["loss"])
            train_acc_hist.append(train_metrics["accuracy"])
            val_loss_hist.append(val_metrics["loss"])
            val_acc_hist.append(val_metrics["accuracy"])
            
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_val_metric=best_val_loss,
                config=self.config,
                is_best=is_best,
            )
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
            
            self.early_stopping(val_metrics["loss"])
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        logger.info("Training pipeline completed.")
        
        return TrainingHistory(
            train_loss=train_loss_hist,
            val_loss=val_loss_hist,
            train_acc=train_acc_hist,
            val_acc=val_acc_hist,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
