"""
Registry and implementations for experiment comparison plotting.
"""
import logging
from pathlib import Path
from typing import List, Callable, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

from training.experiment_management.models import Experiment

logger = logging.getLogger(__name__)

# Type for plotting functions
PlotFunction = Callable[[List[Experiment], Path], None]

# Global Plot Registry
_PLOT_REGISTRY: Dict[str, PlotFunction] = {}

def register_plot(name: str):
    """Decorator to register a new plotting function."""
    def wrapper(func: PlotFunction):
        _PLOT_REGISTRY[name] = func
        return func
    return wrapper


class PlotGenerator:
    """Generates all registered comparison plots."""
    
    @staticmethod
    def generate_plots(experiments: List[Experiment], output_dir: Path) -> None:
        """Executes all registered plotting functions."""
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for name, plot_func in _PLOT_REGISTRY.items():
            try:
                plot_func(experiments, plots_dir)
            except Exception as e:
                logger.error(f"Failed to generate plot '{name}': {e}")


# ==============================================================================
# Registered Plots
# ==============================================================================

def _extract_metric_for_plot(experiments: List[Experiment], metric_type: str, key: str) -> Tuple[List[str], List[float]]:
    names = []
    values = []
    
    for exp in experiments:
        if metric_type == "metrics":
            val = exp.metrics.get(key, "N/A")
        elif metric_type == "deployment":
            val = exp.deployment.get(key, "N/A")
        elif metric_type == "hardware":
            val = exp.hardware.get(key, "N/A")
        else:
            val = "N/A"
            
        if val != "N/A":
            names.append(exp.name)
            values.append(float(val))
            
    return names, values


def _create_bar_chart(names: List[str], values: List[float], title: str, ylabel: str, out_path: Path):
    if not names:
        return
        
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(names))
    bars = plt.bar(x_pos, values, color='skyblue', edgecolor='black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x_pos, names, rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4g}',
                 ha='center', va='bottom', fontsize=9)
                 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@register_plot("f1_comparison")
def plot_f1(experiments: List[Experiment], output_dir: Path):
    names, values = _extract_metric_for_plot(experiments, "metrics", "f1")
    _create_bar_chart(names, values, "F1 Score Comparison", "F1 Score", output_dir / "f1_comparison.png")


@register_plot("accuracy_comparison")
def plot_accuracy(experiments: List[Experiment], output_dir: Path):
    names, values = _extract_metric_for_plot(experiments, "metrics", "accuracy")
    _create_bar_chart(names, values, "Accuracy Comparison", "Accuracy", output_dir / "accuracy_comparison.png")


@register_plot("training_time_comparison")
def plot_training_time(experiments: List[Experiment], output_dir: Path):
    names, values = _extract_metric_for_plot(experiments, "metrics", "training_time")
    _create_bar_chart(names, values, "Training Time Comparison", "Seconds", output_dir / "training_time_comparison.png")


@register_plot("latency_comparison")
def plot_latency(experiments: List[Experiment], output_dir: Path):
    names, values = _extract_metric_for_plot(experiments, "deployment", "latency_ms")
    _create_bar_chart(names, values, "Deployment Latency Comparison", "Latency (ms)", output_dir / "latency_comparison.png")


@register_plot("model_size_comparison")
def plot_model_size(experiments: List[Experiment], output_dir: Path):
    names, values = _extract_metric_for_plot(experiments, "deployment", "size_mb")
    _create_bar_chart(names, values, "Model Size Comparison", "Size (MB)", output_dir / "model_size_comparison.png")
