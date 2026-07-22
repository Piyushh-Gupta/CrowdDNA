"""
CrowdFlow DNA — Baseline Report Generator
=========================================
Module: training/generate_baseline_report.py

Automatically generates the canonical baseline_results.md and baseline.lock
after a successful pipeline execution.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from training.utils import _get_git_commit

logger = logging.getLogger(__name__)


def compute_md5(file_path: Path) -> str:
    if not file_path.exists():
        return "not_found"
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_git_tag() -> str:
    import subprocess
    try:
        tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"]).strip().decode("utf-8")
        return tag
    except Exception:
        return "unknown"


def generate_report_and_lock(
    run_dir: Path,
    dataset_dir: Path,
    dataset_stats_json: Path,
    deployment_stats: dict
) -> None:
    logger.info("Generating baseline_results.md and baseline.lock...")
    
    # Load dataset stats
    with open(dataset_stats_json, "r", encoding="utf-8") as f:
        ds = json.load(f)
        
    # Load training metadata
    with open(run_dir / "metadata" / "experiment_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    # Load evaluation metrics
    with open(run_dir / "metrics" / "evaluation_metrics.json", "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)
        
    # Load training history
    with open(run_dir / "metrics" / "training_history.json", "r", encoding="utf-8") as f:
        history = json.load(f)
        
    # Prepare template values
    values = {}
    
    # 1. Dataset
    values["dataset_path"] = str(dataset_dir)
    values["total_sequences"] = ds.get("total_sequences", 0)
    values["total_frames"] = ds.get("total_frames", 0)
    values["average_sequence_length"] = round(ds.get("average_sequence_length", 0), 2)
    values["min_sequence_length"] = ds.get("min_sequence_length", 0)
    values["max_sequence_length"] = ds.get("max_sequence_length", 0)
    values["average_nodes_per_frame"] = round(ds.get("average_nodes_per_frame", 0), 2)
    values["average_edges_per_frame"] = round(ds.get("average_edges_per_frame", 0), 2)
    
    class_dist = ds.get("class_distribution", {})
    values["class_safe"] = class_dist.get("Safe", 0)
    values["class_congested"] = class_dist.get("Congested", 0)
    values["class_critical"] = class_dist.get("Critical", 0)
    
    values["empty_graph_count"] = ds.get("empty_graph_count", 0)
    values["invalid_graph_count"] = ds.get("invalid_graph_count", 0)
    values["missing_labels"] = ds.get("missing_labels", 0)
    
    # 2. Training
    epochs_run = len(history.get("train_losses", []))
    values["epochs"] = epochs_run
    values["best_epoch"] = history.get("best_epoch", 0)
    # Total training time might not be in history, let's check report if available.
    # We'll calculate it from metadata if present, else placeholder.
    # Meta has it if we parse report_meta but ReportGenerator has it. We'll leave it as unknown if missing.
    values["total_training_time"] = "See run_baseline logs" 
    
    # 3. Evaluation
    values["accuracy"] = round(eval_metrics.get("accuracy", 0), 4)
    values["precision_macro"] = round(eval_metrics.get("precision_macro", 0), 4)
    values["recall_macro"] = round(eval_metrics.get("recall_macro", 0), 4)
    values["f1_macro"] = round(eval_metrics.get("f1_macro", 0), 4)
    
    f1_classes = eval_metrics.get("f1_per_class", [0.0, 0.0, 0.0])
    values["f1_safe"] = round(f1_classes[0], 4) if len(f1_classes) > 0 else 0
    values["f1_congested"] = round(f1_classes[1], 4) if len(f1_classes) > 1 else 0
    values["f1_critical"] = round(f1_classes[2], 4) if len(f1_classes) > 2 else 0
    
    # 4. Deployment
    values.update(deployment_stats)
    
    # 5. Environment
    git_commit = meta.get("git_commit", _get_git_commit())
    git_tag = _get_git_tag()
    values["git_commit"] = git_commit
    values["git_tag"] = git_tag
    cfg = meta.get("configuration_snapshot", {})
    values["configuration_snapshot"] = yaml.dump(cfg, default_flow_style=False)
    
    # Read Template
    with open("docs/BASELINE_RESULTS_TEMPLATE.md", "r", encoding="utf-8") as f:
        template = f.read()
        
    result_md = template.format(**values)
    
    with open("baseline_results.md", "w", encoding="utf-8") as f:
        f.write(result_md)
        
    logger.info("Saved baseline_results.md")
    
    # Generate baseline.lock
    lock_data = {
        "git_tag": git_tag,
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration_hash": compute_md5(run_dir / "config" / "snapshot.yaml"),
        "dataset_manifest_hash": compute_md5(dataset_dir / "manifest.json"),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__
    }
    
    with open(run_dir / "baseline.lock", "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
        
    logger.info(f"Saved baseline.lock to {run_dir}")
