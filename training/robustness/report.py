"""
CrowdFlow DNA — Report Builder
==============================
Module: training/robustness/report.py

Generates JSON, CSV, and Markdown reports for evaluation results.
"""

from __future__ import annotations

import csv
import json
import logging
from typing import Dict

from training.robustness.context import EvaluationContext
from training.robustness import plotting

logger = logging.getLogger("crowddna.robustness")


class ReportBuilder:
    """Assembles metrics into file-based reports."""
    
    def __init__(self, context: EvaluationContext, results: Dict[str, Dict[str, float]]) -> None:
        self.context = context
        self.results = results
        self.protocol_dir = self.context.get_output_path(self.context.protocol_name)
        self.protocol_dir.mkdir(parents=True, exist_ok=True)
        
    def build(self) -> None:
        """Generates all reports."""
        self._write_json()
        self._write_csv()
        self._write_markdown()
        self._generate_plots()
        logger.info(f"Reports saved to {self.protocol_dir}")
        
    def _write_json(self) -> None:
        path = self.protocol_dir / "results.json"
        
        payload = {
            "metadata": {
                "protocol": self.context.protocol_name,
                "deployment_model": str(self.context.deployment_model_path),
                "random_seed": self.context.random_seed
            },
            "results": self.results
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
    def _write_csv(self) -> None:
        path = self.protocol_dir / "results.csv"
        
        if not self.results:
            return
            
        # Get metrics names from the first scenario
        first_scenario = list(self.results.keys())[0]
        metric_names = list(self.results[first_scenario].keys())
        
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario"] + metric_names)
            
            for scenario, metrics in self.results.items():
                row = [scenario] + [metrics.get(m, 0.0) for m in metric_names]
                writer.writerow(row)
                
    def _write_markdown(self) -> None:
        path = self.protocol_dir / "report.md"
        
        md = [
            f"# {self.context.protocol_name.capitalize()} Evaluation Report",
            "",
            "## Configuration",
            f"- **Model**: `{self.context.deployment_model_path}`",
            f"- **Seed**: `{self.context.random_seed}`",
            "",
            "## Results",
            ""
        ]
        
        if not self.results:
            md.append("No results to display.")
        else:
            first_scenario = list(self.results.keys())[0]
            metric_names = list(self.results[first_scenario].keys())
            
            header = "| Scenario | " + " | ".join(metric_names) + " |"
            sep = "| --- | " + " | ".join(["---"] * len(metric_names)) + " |"
            md.extend([header, sep])
            
            for scenario, metrics in self.results.items():
                row = f"| {scenario} | " + " | ".join([f"{metrics.get(m, 0.0):.4f}" for m in metric_names]) + " |"
                md.append(row)
                
        md.extend([
            "",
            "## Plots",
            "![Degradation](plots/degradation.png)",
            "![Latency](plots/latency.png)",
        ])
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
            
    def _generate_plots(self) -> None:
        plots_dir = self.protocol_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plotting.plot_degradation(self.results, plots_dir / "degradation.png")
        plotting.plot_latency(self.results, plots_dir / "latency.png")
