"""
CrowdFlow DNA — Evaluation Reporting & Visualization
==================================================
Module: training/reporting.py

Implements the reporting layer that converts EvaluationResult into reusable
artifacts for experiment analysis.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from training.evaluate_model import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportArtifacts:
    """Strictly typed container for all generated reporting artifacts."""
    confusion_matrix_png: Path
    normalized_confusion_matrix_png: Path
    roc_curve_png: Path
    precision_recall_curve_png: Path
    metrics_csv: Path
    metrics_json: Path
    summary_markdown: Path


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


class ReportGenerator:
    """Generates evaluation reports and standard ML visualisations."""
    
    def __init__(self, output_dir: str | Path, class_names: list[str]) -> None:
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def _generate_confusion_matrix(self, result: EvaluationResult) -> tuple[Path, Path]:
        cm = result.confusion_matrix
        cm_norm = result.normalized_confusion_matrix
        
        # 1. Raw Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="left")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, str(cm[i, j]), va="center", ha="center")
                
        raw_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(raw_path, bbox_inches="tight")
        plt.close(fig)
        
        # 2. Normalized Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm_norm, cmap="Blues")
        fig.colorbar(cax)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="left")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", va="center", ha="center")
                
        norm_path = self.output_dir / "normalized_confusion_matrix.png"
        fig.savefig(norm_path, bbox_inches="tight")
        plt.close(fig)
        
        return raw_path, norm_path

    def _generate_roc_curve(self, result: EvaluationResult) -> Path:
        targets = result.targets
        probabilities = result.probabilities
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, class_name in enumerate(self.class_names):
            binary_targets = (targets == i).astype(int)
            if binary_targets.sum() == 0:
                logger.warning(
                    f"Class '{class_name}' has no positive samples. Skipping ROC curve."
                )
                continue
                
            fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
            
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        
        out_path = self.output_dir / "roc_curve.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _generate_pr_curve(self, result: EvaluationResult) -> Path:
        targets = result.targets
        probabilities = result.probabilities
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, class_name in enumerate(self.class_names):
            binary_targets = (targets == i).astype(int)
            if binary_targets.sum() == 0:
                logger.warning(
                    f"Class '{class_name}' has no positive samples. Skipping PR curve."
                )
                continue
                
            precision, recall, _ = precision_recall_curve(binary_targets, probabilities[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f"{class_name} (AUC = {pr_auc:.2f})")
            
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        
        out_path = self.output_dir / "precision_recall_curve.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _generate_metrics_table(self, result: EvaluationResult) -> Path:
        out_path = self.output_dir / "metrics.csv"
        
        support = result.confusion_matrix.sum(axis=1)
        
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
            for i, class_name in enumerate(self.class_names):
                writer.writerow([
                    class_name,
                    f"{result.precision_per_class[i]:.4f}",
                    f"{result.recall_per_class[i]:.4f}",
                    f"{result.f1_per_class[i]:.4f}",
                    int(support[i])
                ])
                
        return out_path
        
    def _generate_metrics_json(self, result: EvaluationResult) -> Path:
        import dataclasses
        out_path = self.output_dir / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, cls=NumpyEncoder, indent=2)
        return out_path
        
    def _generate_markdown(
        self,
        result: EvaluationResult,
        metadata: dict[str, Any],
        artifacts: dict[str, Path]
    ) -> Path:
        out_path = self.output_dir / "report.md"
        
        timestamp = metadata.get("timestamp", "N/A")
        git_commit = metadata.get("git_commit", "None")
        train_time = metadata.get("total_training_time", "N/A")
        eval_time = metadata.get("evaluation_time", "N/A")
        best_ckpt = metadata.get("best_checkpoint_path", "N/A")
        
        cfg_str = json.dumps(metadata.get("configuration_snapshot", {}), indent=2)
        
        md_content = f"""# Evaluation Report

## Experiment Metadata
- **Timestamp**: {timestamp}
- **Git Commit**: {git_commit}
- **Total Training Time**: {train_time}
- **Evaluation Time**: {eval_time}
- **Best Checkpoint**: {best_ckpt}

## Model Configuration
```yaml
{cfg_str}
```

## Global Metrics
- **Accuracy**: {result.accuracy:.4f}
- **Macro Precision**: {result.precision_macro:.4f}
- **Macro Recall**: {result.recall_macro:.4f}
- **Macro F1**: {result.f1_macro:.4f}

## Metrics Table
[Download CSV](metrics.csv)

## Visualizations
### Confusion Matrix
![Confusion Matrix]({artifacts['cm'].name})

### Normalized Confusion Matrix
![Normalized Confusion Matrix]({artifacts['norm_cm'].name})

### ROC Curve
![ROC Curve]({artifacts['roc'].name})

### Precision-Recall Curve
![Precision-Recall Curve]({artifacts['pr'].name})
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return out_path

    def generate(self, result: EvaluationResult, metadata: dict[str, Any]) -> ReportArtifacts:
        """Generates all reporting artifacts from the evaluation outcomes."""
        if len(result.probabilities) == 0:
            raise ValueError("Empty probability matrix provided to ReportGenerator.")
            
        cm_path, norm_cm_path = self._generate_confusion_matrix(result)
        roc_path = self._generate_roc_curve(result)
        pr_path = self._generate_pr_curve(result)
        csv_path = self._generate_metrics_table(result)
        json_path = self._generate_metrics_json(result)
        
        artifacts_dict = {
            "cm": cm_path,
            "norm_cm": norm_cm_path,
            "roc": roc_path,
            "pr": pr_path
        }
        md_path = self._generate_markdown(result, metadata, artifacts_dict)
        
        return ReportArtifacts(
            confusion_matrix_png=cm_path,
            normalized_confusion_matrix_png=norm_cm_path,
            roc_curve_png=roc_path,
            precision_recall_curve_png=pr_path,
            metrics_csv=csv_path,
            metrics_json=json_path,
            summary_markdown=md_path
        )
