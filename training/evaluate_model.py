"""
CrowdFlow DNA — Evaluation Framework
====================================
Module: training/evaluate_model.py

Implements a comprehensive evaluation framework for trained CrowdDNA models.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset
from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig
from training.train_model import CheckpointManager, sequence_collate_fn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Strictly typed container for all evaluation metrics and outputs."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_per_class: np.ndarray
    recall_per_class: np.ndarray
    f1_per_class: np.ndarray
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray


class MetricsComputer:
    """Computes all classification metrics and confusion matrices."""
    
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def compute(
        self, targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> EvaluationResult:
        if len(targets) == 0:
            raise ValueError("Empty targets provided to MetricsComputer.")
            
        accuracy = accuracy_score(targets, predictions)
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average="macro", zero_division=0
        )
        
        precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(
            targets, predictions, average=None, labels=range(self.num_classes), zero_division=0
        )
        
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        cm_norm = confusion_matrix(
            targets, predictions, labels=range(self.num_classes), normalize="true"
        )
        
        return EvaluationResult(
            accuracy=float(accuracy),
            precision_macro=float(precision_macro),
            recall_macro=float(recall_macro),
            f1_macro=float(f1_macro),
            precision_per_class=precision_cls,
            recall_per_class=recall_cls,
            f1_per_class=f1_cls,
            confusion_matrix=cm,
            normalized_confusion_matrix=cm_norm,
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )


class EvaluationEngine:
    """Engine responsible for loading checkpoints and running inference."""
    
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        
        model_cfg = CrowdDNAModelConfig.from_dict(config.get("model", {}))
        self.model = CrowdDNAModel(model_cfg).to(self.device)
        self.num_classes = model_cfg.num_classes
        self.metrics_computer = MetricsComputer(self.num_classes)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restores model weights using the CheckpointManager."""
        checkpoint_dir = os.path.dirname(checkpoint_path)
        manager = CheckpointManager(checkpoint_dir)
        # We only need the model state for evaluation
        manager.load(checkpoint_path, self.model)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
    def evaluate(self, dataset: SequenceGraphDataset, batch_size: int = 4) -> EvaluationResult:
        """Runs inference on the dataset and computes all metrics."""
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Cannot evaluate.")
            
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            raise ValueError(
                f"Device mismatch: Model is on {model_device} but engine expects {self.device}"
            )
            
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )
        
        self.model.eval()
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                sequences_on_device = [
                    [data.to(self.device) for data in seq] for seq in batch.sequences
                ]
                labels = batch.labels.to(self.device)
                
                logits = self.model(sequences_on_device)
                all_logits.append(logits)
                all_targets.append(labels)
                
        if not all_logits:
            raise ValueError("No predictions were generated (dataset might be corrupted).")
            
        logits_tensor = torch.cat(all_logits, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        
        probabilities = torch.softmax(logits_tensor, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        
        # Verify shapes before metric computation
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(
                f"Shape violation: Expected probabilities (N, {self.num_classes}), "
                f"got {probabilities.shape}"
            )
            
        if predictions.shape != targets_tensor.shape:
            raise ValueError(
                f"Shape violation: predictions {predictions.shape} vs targets {targets_tensor.shape}"
            )
            
        return self.metrics_computer.compute(
            targets=targets_tensor.cpu().numpy(),
            predictions=predictions.cpu().numpy(),
            probabilities=probabilities.cpu().numpy(),
        )
