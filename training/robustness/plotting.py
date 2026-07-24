"""
CrowdFlow DNA — Plotting
========================
Module: training/robustness/plotting.py

Generates visualizations for robustness and stress evaluations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("crowddna.robustness")

def plot_degradation(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Plots a bar chart comparing performance across scenarios."""
    scenarios = list(results.keys())
    accuracies = [results[s].get("Accuracy", 0.0) for s in scenarios]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(scenarios))
    plt.bar(x, accuracies, color='skyblue', edgecolor='black')
    
    plt.axhline(y=accuracies[0], color='red', linestyle='--', label='Baseline')
    
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.ylabel("Accuracy")
    plt.title("Robustness Performance Degradation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_latency(results: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Plots a bar chart comparing latency across scenarios."""
    scenarios = list(results.keys())
    latencies = [results[s].get("Average Latency (ms)", 0.0) for s in scenarios]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(scenarios))
    plt.bar(x, latencies, color='salmon', edgecolor='black')
    
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency across Scenarios")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
