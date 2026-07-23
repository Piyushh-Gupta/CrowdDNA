"""
Generates CSV, JSON, and Markdown leaderboard outputs.
"""
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from training.experiment_management.models import Experiment

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"

class LeaderboardGenerator:
    """Generates leaderboard outputs in various formats."""

    @staticmethod
    def _get_metadata() -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION
        }

    @staticmethod
    def _flatten_experiment(exp: Experiment) -> dict:
        """Flattens experiment data into a flat dict for CSV/JSON."""
        return {
            "Rank": exp.rank if exp.rank is not None else "N/A",
            "Experiment": exp.name,
            "Timestamp": exp.timestamp,
            "Git Commit": exp.git_commit,
            "F1 Score": exp.metrics.get("f1", "N/A"),
            "Accuracy": exp.metrics.get("accuracy", "N/A"),
            "Precision": exp.metrics.get("precision", "N/A"),
            "Recall": exp.metrics.get("recall", "N/A"),
            "Deployment Latency (ms)": exp.deployment.get("latency_ms", "N/A"),
            "TorchScript Size (MB)": exp.deployment.get("size_mb", "N/A"),
            "Training Time (s)": exp.metrics.get("training_time", "N/A"),
            "Peak GPU Alloc (MB)": exp.hardware.get("peak_gpu_allocated_mb", "N/A"),
            "Path": exp.path
        }

    @staticmethod
    def generate_json(experiments: List[Experiment], output_dir: Path) -> None:
        out_path = output_dir / "leaderboard.json"
        
        payload = {
            "_meta": LeaderboardGenerator._get_metadata(),
            "leaderboard": [LeaderboardGenerator._flatten_experiment(e) for e in experiments]
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            
    @staticmethod
    def generate_csv(experiments: List[Experiment], output_dir: Path) -> None:
        out_path = output_dir / "leaderboard.csv"
        
        if not experiments:
            return
            
        flat_data = [LeaderboardGenerator._flatten_experiment(e) for e in experiments]
        headers = list(flat_data[0].keys())
        
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(flat_data)

    @staticmethod
    def generate_markdown(experiments: List[Experiment], output_dir: Path) -> None:
        out_path = output_dir / "leaderboard.md"
        meta = LeaderboardGenerator._get_metadata()
        
        lines = [
            "# CrowdDNA Experiment Leaderboard\n",
            f"*Generated on {meta['generation_timestamp']} (Schema v{meta['schema_version']})*\n",
            "---\n"
        ]
        
        if not experiments:
            lines.append("No valid experiments found.\n")
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return

        best = experiments[0]
        
        lines.extend([
            "## 🏆 Best Overall Model\n",
            f"**Experiment:** `{best.name}`\n",
            f"**Path:** `{best.path}`\n",
            f"- **F1 Score:** {best.metrics.get('f1', 'N/A')}\n",
            f"- **Accuracy:** {best.metrics.get('accuracy', 'N/A')}\n",
            f"- **Deployment Latency:** {best.deployment.get('latency_ms', 'N/A')} ms\n",
            "\n---\n"
        ])
        
        lines.append("## Leaderboard\n")
        
        flat_data = [LeaderboardGenerator._flatten_experiment(e) for e in experiments]
        headers = ["Rank", "Experiment", "F1 Score", "Accuracy", "Deployment Latency (ms)", "Training Time (s)", "Path"]
        
        # Table Header
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines.extend([header_row + "\n", sep_row + "\n"])
        
        for row in flat_data:
            vals = [str(row.get(h, "N/A")) for h in headers]
            lines.append("| " + " | ".join(vals) + " |\n")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
