"""
CrowdFlow DNA — Experiment Runner
=================================
Module: training/run_experiment.py

Implements a reproducible experiment execution framework combining training
and evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.evaluate_model import EvaluationEngine, EvaluationResult
from training.reporting import ReportArtifacts, ReportGenerator
from training.train_model import Trainer, TrainingHistory, set_random_seed

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy arrays and scalars into JSON-serializable formats."""
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return super().default(o)


class ArtifactManager:
    """Manages persistence of all experiment artifacts."""
    
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.metrics_dir = self.base_dir / "metrics"
        self.metadata_dir = self.base_dir / "metadata"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        
        self._create_layout()
        
    def _create_layout(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def save_config(self, config: dict[str, Any]) -> None:
        with open(self.config_dir / "snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)
            
    def save_training_history(self, history: TrainingHistory) -> None:
        # TODO: Learning curves extension point
        with open(self.metrics_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(history), f, cls=NumpyEncoder, indent=2)
            
    def save_evaluation_metrics(self, result: EvaluationResult) -> None:
        # TODO: ROC curves extension point
        # TODO: PR curves extension point
        with open(self.metrics_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, cls=NumpyEncoder, indent=2)
            
    def save_metadata(self, metadata: dict[str, Any]) -> None:
        with open(self.metadata_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, cls=NumpyEncoder, indent=2)


@dataclass(frozen=True)
class ExperimentResult:
    """Structured container for experiment outcomes."""
    training_history: TrainingHistory
    evaluation_result: EvaluationResult
    best_checkpoint_path: str
    total_training_time: float
    evaluation_time: float
    configuration_snapshot: dict[str, Any]
    git_commit_hash: str | None
    timestamp: str
    experiment_directory: Path
    report_artifacts: ReportArtifacts | None = None


class ExperimentRunner:
    """Executes a complete, reproducible experiment lifecycle."""
    
    def __init__(self, config_path: str, experiment_dir: str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.artifact_manager = ArtifactManager(self.experiment_dir)
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        # Guarantee checkpoint overrides correctly point inside the experiment layout
        if "training" not in self.config:
            self.config["training"] = {}
        self.config["training"]["checkpoint_dir"] = str(self.artifact_manager.checkpoints_dir)
        
        self.seed = self.config["training"].get("random_seed", 42)
        deterministic = self.config["training"].get("deterministic", False)
        set_random_seed(self.seed, deterministic)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _get_git_commit(self) -> str | None:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return res.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
            
    def run(self) -> ExperimentResult:
        logger.info(f"Starting experiment in {self.experiment_dir}")
        timestamp = datetime.now(timezone.utc).isoformat()
        commit_hash = self._get_git_commit()
        
        # 1. Preparation
        self.artifact_manager.save_config(self.config)
        metadata = {
            "timestamp": timestamp,
            "git_commit": commit_hash,
            "random_seed": self.seed,
            "configuration_snapshot": self.config,
        }
        self.artifact_manager.save_metadata(metadata)
        
        # 2. Training Phase
        logger.info("Initializing Trainer...")
        # Point the Trainer at the newly saved, overridden configuration
        snapshot_path = self.artifact_manager.config_dir / "snapshot.yaml"
        trainer = Trainer(str(snapshot_path))
        
        start_train = time.perf_counter()
        training_history = trainer.fit()
        train_time = time.perf_counter() - start_train
        
        self.artifact_manager.save_training_history(training_history)
        
        best_ckpt = str(self.artifact_manager.checkpoints_dir / "best.pt")
        
        # TODO: ONNX export extension point
        
        # 3. Evaluation Phase
        logger.info("Initializing Evaluation Engine...")
        eval_engine = EvaluationEngine(self.config, self.device)
        eval_engine.load_checkpoint(best_ckpt)
        
        start_eval = time.perf_counter()
        evaluation_result = eval_engine.evaluate(trainer.val_dataset)
        eval_time = time.perf_counter() - start_eval
        
        self.artifact_manager.save_evaluation_metrics(evaluation_result)
        
        # 4. Generate Reports
        logger.info("Generating evaluation reports...")
        report_gen = ReportGenerator(
            self.artifact_manager.reports_dir, self.config["model"]["classes"]
        )
        report_meta = {
            "timestamp": timestamp,
            "git_commit": commit_hash,
            "total_training_time": train_time,
            "evaluation_time": eval_time,
            "best_checkpoint_path": best_ckpt,
            "configuration_snapshot": self.config,
        }
        report_artifacts = report_gen.generate(evaluation_result, report_meta)
        
        logger.info("Experiment successfully completed.")
        return ExperimentResult(
            training_history=training_history,
            evaluation_result=evaluation_result,
            best_checkpoint_path=best_ckpt,
            total_training_time=train_time,
            evaluation_time=eval_time,
            configuration_snapshot=self.config,
            git_commit_hash=commit_hash,
            timestamp=timestamp,
            experiment_directory=self.experiment_dir,
            report_artifacts=report_artifacts,
        )
