"""
CrowdFlow DNA — Baseline Experiment Workflow
============================================
Module: training/run_baseline.py

Implements the canonical workflow used to execute the project's baseline experiment.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from crowdflow_dna.model.crowddna_model import CrowdDNAModel
from training.run_experiment import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "experiments" / "baseline.yaml"
        
        if not config_path.exists():
            logger.error(f"Baseline configuration not found at {config_path}")
            return 1
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = project_root / "experiments" / "runs" / f"baseline_{timestamp}"
        
        logger.info("Initializing Baseline Workflow...")
        logger.info(f"Target Directory: {exp_dir}")
        
        runner = ExperimentRunner(str(config_path), str(exp_dir))
        
        logger.info("Executing ExperimentRunner...")
        result = runner.run()
        
        # Append specific metadata requested for the baseline
        metadata_file = exp_dir / "metadata" / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                
            model = CrowdDNAModel(cfg["model"])
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metadata["model_architecture"] = "CrowdDNAModel"
            metadata["dataset_version"] = "synthetic"
            metadata["total_parameters"] = total_params
            metadata["trainable_parameters"] = trainable_params
            metadata["training_duration"] = result.total_training_time
            metadata["evaluation_duration"] = result.evaluation_time
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        # Summary
        summary = f"""
------------------------------------
CrowdDNA Baseline Workflow Complete
------------------------------------

Best Checkpoint: {result.best_checkpoint_path}
Training Time: {result.total_training_time:.2f}s
Evaluation Time: {result.evaluation_time:.2f}s

Accuracy: {result.evaluation_result.accuracy:.4f}
Precision: {result.evaluation_result.precision_macro:.4f}
Recall: {result.evaluation_result.recall_macro:.4f}
F1: {result.evaluation_result.f1_macro:.4f}

Experiment Directory: {result.experiment_directory}
------------------------------------
"""
        print(summary)
        return 0
        
    except Exception:
        logger.exception("Baseline workflow failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
