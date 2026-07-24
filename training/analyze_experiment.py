"""
CrowdFlow DNA — General Experiment Analysis Framework
=====================================================
Module: training/analyze_experiment.py

Consumes a completed experiment directory, calculates statistics, 
generates plots (reusing existing ones where possible), and produces 
a comprehensive, self-contained benchmark analysis report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Try to import torch for environment/metadata extraction
try:
    import torch
    import torch_geometric
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


def compute_descriptive_stats(data: List[float]) -> Dict[str, float]:
    """Computes basic descriptive statistics for a list of values."""
    if not data:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95))
    }


def compute_prediction_entropy(probabilities: List[List[float]]) -> float:
    """Computes the average prediction entropy across the dataset."""
    if not probabilities:
        return 0.0
    arr = np.array(probabilities)
    # Clip to avoid log(0)
    arr = np.clip(arr, 1e-9, 1.0)
    entropy = -np.sum(arr * np.log2(arr), axis=1)
    return float(np.mean(entropy))


def get_gpu_memory(exp_dir: Path) -> str:
    """Retrieves peak GPU memory using priority logic."""
    # 1. Check experiment metadata
    meta_path = exp_dir / "metadata" / "experiment_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "peak_gpu_memory" in meta:
                return f"{meta['peak_gpu_memory']} MB (Experiment Metadata)"
    
    # 2. Check deployment validation
    dep_val_md = exp_dir.parent.parent / "deployment_validation" / "deployment_validation.md"
    if dep_val_md.exists():
        with open(dep_val_md, "r", encoding="utf-8") as f:
            content = f.read()
            # If we logged it here, we could parse it
            match = re.search(r"\*\*Peak GPU Memory\*\*: ([\d.]+) MB", content)
            if match:
                return f"{match.group(1)} MB (Deployment Validation Report)"
                
    # 3. Training logs 
    log_path = exp_dir / "logs" / "training.log"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"Peak GPU memory.*?([\d.]+)\s*MB", line, re.IGNORECASE)
                if match:
                    return f"{match.group(1)} MB (Training Logs)"
    
    # 4. Current torch.cuda values (if running locally and GPU is available)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        if allocated > 0 or reserved > 0:
            return f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB (Current torch.cuda)"
            
    # 5. Fallback
    return "N/A"


def capture_environment() -> Dict[str, str]:
    """Captures system environment information."""
    env = {
        "Python": sys.version.split(" ")[0],
        "Platform": platform.platform(),
        "CPU": platform.processor() or "Unknown",
        "System RAM": "N/A (psutil not guaranteed)",
        "PyTorch": torch.__version__ if TORCH_AVAILABLE else "N/A",
        "CUDA": torch.version.cuda if (TORCH_AVAILABLE and torch.version.cuda) else "N/A",
        "Torch Geometric": torch_geometric.__version__ if TORCH_AVAILABLE else "N/A",
        "GPU Name": torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else "N/A",
    }
    return env


def plot_loss_curve(history: dict, out_path: Path) -> None:
    """Plots training and validation loss."""
    if "train_loss" not in history or "val_loss" not in history:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curve(history: dict, out_path: Path) -> None:
    """Plots training and validation accuracy."""
    if "train_acc" not in history or "val_acc" not in history:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history["train_acc"], label="Train Accuracy")
    ax.plot(history["val_acc"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(probs: List[List[float]], out_path: Path) -> None:
    """Plots histogram of maximum confidence scores."""
    if not probs:
        return
    max_probs = [max(p) for p in probs]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(max_probs, bins=20, range=(0.0, 1.0), edgecolor="black")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(predictions: List[int], targets: List[int], classes: List[str], out_path: Path) -> None:
    """Plots bar chart of predicted vs actual class distribution."""
    if not predictions or not targets:
        return
        
    pred_counts = [predictions.count(i) for i in range(len(classes))]
    target_counts = [targets.count(i) for i in range(len(classes))]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, target_counts, width, label='Actual')
    ax.bar(x + width/2, pred_counts, width, label='Predicted')
    
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution (Actual vs Predicted)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def extract_latency(exp_dir: Path) -> Any:
    dep_val_md = exp_dir.parent.parent / "deployment_validation" / "deployment_validation.md"
    if dep_val_md.exists():
        with open(dep_val_md, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"\*\*TorchScript via InferenceRuntime\*\*: ([\d.]+)\s*ms", content)
            if match:
                return float(match.group(1))
    return "N/A"
    
def extract_model_sizes(exp_dir: Path) -> tuple[Any, Any]:
    ts_size = "N/A"
    onnx_size = "N/A"
    dep_val_md = exp_dir.parent.parent / "deployment_validation" / "deployment_validation.md"
    if dep_val_md.exists():
        with open(dep_val_md, "r", encoding="utf-8") as f:
            content = f.read()
            match_ts = re.search(r"\*\*TorchScript\*\*: ([\d.]+)\s*MB", content)
            match_onnx = re.search(r"\*\*ONNX\*\*: ([\d.]+)\s*MB", content)
            if match_ts:
                ts_size = float(match_ts.group(1))
            if match_onnx:
                onnx_size = float(match_onnx.group(1))
    return ts_size, onnx_size


def analyze_experiment(exp_dir: Path) -> None:
    if not exp_dir.exists() or not exp_dir.is_dir():
        logger.error(f"Experiment directory {exp_dir} does not exist.")
        sys.exit(1)

    # Output Structure
    analysis_dir = exp_dir / "analysis"
    report_dir = analysis_dir / "report"
    plots_dir = analysis_dir / "plots"
    tables_dir = analysis_dir / "tables"
    raw_dir = analysis_dir / "raw"

    for d in [report_dir, plots_dir, tables_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Read Inputs
    history_file = exp_dir / "metrics" / "training_history.json"
    eval_file = exp_dir / "metrics" / "evaluation_metrics.json"
    meta_file = exp_dir / "metadata" / "experiment_metadata.json"

    history = {}
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

    eval_metrics = {}
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    meta = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 2. Extract Data
    config = meta.get("configuration_snapshot", {})
    model_config = config.get("model", {})
    classes = model_config.get("classes", ["Safe", "Congesting", "Critical"])
    
    probs = eval_metrics.get("probabilities", [])
    max_probs = [max(p) for p in probs] if probs else []
    
    # 3. Compute Statistics
    conf_stats = compute_descriptive_stats(max_probs)
    entropy = compute_prediction_entropy(probs)
    
    # ECE and Brier Score placeholders
    ece = "N/A (Placeholder)"
    brier = "N/A (Placeholder)"

    # Dataset stats
    targets = eval_metrics.get("targets", [])
    val_sample_count = len(targets)
    # Estimate max seq len or get from config
    
    # 4. Generate/Copy Plots
    # Loss
    loss_path = plots_dir / "loss_curve.png"
    if not loss_path.exists():
        plot_loss_curve(history, loss_path)
    
    # Accuracy
    acc_path = plots_dir / "accuracy_curve.png"
    if not acc_path.exists():
        plot_accuracy_curve(history, acc_path)
        
    # Confidence
    conf_path = plots_dir / "confidence_histogram.png"
    if not conf_path.exists():
        plot_confidence_histogram(probs, conf_path)
        
    # Class Distribution
    dist_path = plots_dir / "class_distribution.png"
    if not dist_path.exists():
        plot_class_distribution(eval_metrics.get("predictions", []), targets, classes, dist_path)

    # Existing plots from reports/
    reports_dir = exp_dir / "reports"
    if reports_dir.exists():
        for fname in ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"]:
            src = reports_dir / fname
            dst = plots_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                
    # Model Sizes & Latency
    ts_size, onnx_size = extract_model_sizes(exp_dir)
    latency = extract_latency(exp_dir)
                
    # 5. Build Analysis Output JSON
    analysis_data = {
        "experiment_name": exp_dir.name,
        "environment": capture_environment(),
        "model_metadata": {
            "parameter_count": meta.get("total_parameters", "N/A"),
            "trainable_parameters": meta.get("trainable_parameters", "N/A"),
            "checkpoint_size_mb": "N/A", 
            "torchscript_size_mb": ts_size,
            "onnx_size_mb": onnx_size,
            "configuration": model_config,
            "sequence_length": config.get("training", {}).get("sequence_length", "N/A"),
            "gat_hidden_dim": model_config.get("gnn_hidden_dim", "N/A"),
            "attention_heads": model_config.get("gnn_heads", "N/A"),
            "gru_hidden_size": model_config.get("gru_hidden_dim", "N/A"),
            "gru_layers": model_config.get("gru_num_layers", "N/A"),
            "bidirectional": model_config.get("gru_bidirectional", "N/A")
        },
        "dataset_summary": {
            "training_samples": "N/A", 
            "validation_samples": val_sample_count,
            "test_samples": "N/A",
            "class_distribution": {classes[i]: targets.count(i) for i in range(len(classes))} if targets else "N/A",
            "dataset_hash": meta.get("dataset_hash", "N/A"),
            "manifest_version": meta.get("manifest_version", "N/A")
        },
        "statistics": {
            "confidence": conf_stats,
            "prediction_entropy": entropy,
            "expected_calibration_error": ece,
            "brier_score": brier
        },
        "gpu_memory": get_gpu_memory(exp_dir),
        "performance": {
            "accuracy": eval_metrics.get("accuracy", "N/A"),
            "precision_macro": eval_metrics.get("precision_macro", "N/A"),
            "recall_macro": eval_metrics.get("recall_macro", "N/A"),
            "f1_macro": eval_metrics.get("f1_macro", "N/A"),
            "training_time": meta.get("training_duration", "N/A"),
            "evaluation_time": meta.get("evaluation_duration", "N/A"),
            "latency": latency
        }
    }
    
    # Try to grab checkpoint size
    best_ckpt = exp_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        analysis_data["model_metadata"]["checkpoint_size_mb"] = round(best_ckpt.stat().st_size / (1024 * 1024), 2)
        
    analysis_json_path = raw_dir / "analysis.json"
    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)
        
    # 6. Generate CSV Summary
    csv_path = tables_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", "Accuracy", "Precision", "Recall", "F1", 
            "Latency", "Training Time", "Evaluation Time", "GPU Memory", 
            "Parameter Count", "Checkpoint Size"
        ])
        writer.writerow([
            analysis_data["experiment_name"],
            analysis_data["performance"]["accuracy"],
            analysis_data["performance"]["precision_macro"],
            analysis_data["performance"]["recall_macro"],
            analysis_data["performance"]["f1_macro"],
            analysis_data["performance"]["latency"],
            analysis_data["performance"]["training_time"],
            analysis_data["performance"]["evaluation_time"],
            analysis_data["gpu_memory"],
            analysis_data["model_metadata"]["parameter_count"],
            analysis_data["model_metadata"]["checkpoint_size_mb"]
        ])
        
    # 7. Generate Markdown Report
    md_path = report_dir / "benchmark_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Analysis: {exp_dir.name}\n\n")
        
        f.write("## 1. Model & Environment\n")
        f.write(f"- **Parameters**: {analysis_data['model_metadata']['parameter_count']}\n")
        f.write(f"- **Checkpoint Size**: {analysis_data['model_metadata']['checkpoint_size_mb']} MB\n")
        f.write(f"- **GAT Hidden Dim**: {analysis_data['model_metadata']['gat_hidden_dim']}\n")
        f.write(f"- **GRU Hidden Size**: {analysis_data['model_metadata']['gru_hidden_size']}\n")
        f.write(f"- **GPU Memory**: {analysis_data['gpu_memory']}\n\n")
        
        f.write("## 2. Performance Summary\n")
        f.write(f"- **Accuracy**: {analysis_data['performance']['accuracy']}\n")
        f.write(f"- **F1 Macro**: {analysis_data['performance']['f1_macro']}\n")
        f.write(f"- **Training Time**: {analysis_data['performance']['training_time']}s\n")
        f.write(f"- **TorchScript Latency**: {analysis_data['performance']['latency']}ms\n\n")
        
        f.write("## 3. Statistical Analysis\n")
        f.write(f"- **Mean Confidence**: {conf_stats['mean']:.4f}\n")
        f.write(f"- **Median Confidence**: {conf_stats['median']:.4f}\n")
        f.write(f"- **Prediction Entropy**: {entropy:.4f}\n")
        f.write(f"- **ECE**: {ece}\n")
        f.write(f"- **Brier Score**: {brier}\n\n")
        
        f.write("## 4. Visualizations\n")
        f.write("*(See `plots/` directory for full resolution curves and distributions)*\n")

    logger.info(f"Analysis completed successfully for {exp_dir.name}")
    logger.info(f"Outputs written to {analysis_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze CrowdDNA Experiment")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory")
    args = parser.parse_args()
    analyze_experiment(Path(args.experiment_dir))
