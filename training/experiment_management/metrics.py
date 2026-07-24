"""
Centralized metric extraction module.
"""
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class MetricExtractor:
    """Extracts metrics from various experiment artifacts securely."""

    @staticmethod
    def extract_all(exp_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Extracts all metrics, returning dictionaries for configuration, metrics, deployment, and hardware.
        """
        config = MetricExtractor._extract_metadata(exp_dir)
        metrics = MetricExtractor._extract_analysis(exp_dir)
        hardware, training_time = MetricExtractor._extract_logs(exp_dir)
        deployment = MetricExtractor._extract_deployment(exp_dir)

        if training_time != "N/A":
            metrics["training_time"] = training_time

        return config, metrics, deployment, hardware

    @staticmethod
    def _extract_metadata(exp_dir: Path) -> Dict[str, Any]:
        meta_path = exp_dir / "metadata" / "experiment_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("config", {})
        except Exception as e:
            logger.debug(f"Failed to read metadata for {exp_dir.name}: {e}")
            return {}

    @staticmethod
    def _extract_analysis(exp_dir: Path) -> Dict[str, Any]:
        json_path = exp_dir / "analysis" / "raw" / "analysis.json"
        metrics = {
            "f1": "N/A",
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "roc_auc": "N/A",
            "pr_auc": "N/A"
        }
        
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    perf = data.get("performance", {})
                    metrics["f1"] = perf.get("f1_macro", "N/A")
                    metrics["accuracy"] = perf.get("accuracy", "N/A")
                    metrics["precision"] = perf.get("precision_macro", "N/A")
                    metrics["recall"] = perf.get("recall_macro", "N/A")
                    
                    # We can also get training time from here if logs failed
                    if "training_time" in perf:
                        metrics["training_time"] = perf["training_time"]
                        
                return metrics
            except Exception as e:
                logger.debug(f"Failed to read analysis.json for {exp_dir.name}: {e}")

        # Fallback to CSV
        csv_path = exp_dir / "analysis" / "tables" / "summary.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for k in metrics.keys():
                            if k in row:
                                try:
                                    metrics[k] = float(row[k])
                                except ValueError:
                                    pass
                        break  # only one row expected
            except Exception as e:
                logger.debug(f"Failed to read summary.csv for {exp_dir.name}: {e}")

        return metrics

    @staticmethod
    def _extract_deployment(exp_dir: Path) -> Dict[str, Any]:
        deployment = {
            "latency_ms": "N/A",
            "size_mb": "N/A"
        }
        
        # Check validation markdown report
        report_path = exp_dir / "deploy" / "deployment_validation.md"
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                lat_match = re.search(r"TorchScript via InferenceRuntime\*\*: ([\d.]+) ms", content)
                if lat_match:
                    deployment["latency_ms"] = float(lat_match.group(1))
                    
                size_match = re.search(r"TorchScript\*\*: ([\d.]+) MB", content)
                if size_match:
                    deployment["size_mb"] = float(size_match.group(1))
            except Exception as e:
                logger.debug(f"Failed to parse deployment report for {exp_dir.name}: {e}")
                
        return deployment

    @staticmethod
    def _extract_logs(exp_dir: Path) -> Tuple[Dict[str, Any], Any]:
        hardware = {
            "peak_gpu_allocated_mb": "N/A",
            "peak_gpu_reserved_mb": "N/A"
        }
        training_time = "N/A"
        
        log_path = exp_dir / "logs" / "training.log"
        if not log_path.exists():
            return hardware, training_time
            
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            time_match = re.search(r"Total training time: ([\d.]+)s", content)
            if time_match:
                training_time = float(time_match.group(1))
                
            alloc_match = re.search(r"Peak GPU allocated: ([\d.]+) MB", content)
            if alloc_match:
                hardware["peak_gpu_allocated_mb"] = float(alloc_match.group(1))
                
            res_match = re.search(r"Peak GPU reserved: ([\d.]+) MB", content)
            if res_match:
                hardware["peak_gpu_reserved_mb"] = float(res_match.group(1))
                
        except Exception as e:
            logger.debug(f"Failed to parse training logs for {exp_dir.name}: {e}")
            
        return hardware, training_time
