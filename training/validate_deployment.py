"""
CrowdFlow DNA — Deployment Validation
=====================================
Module: training/validate_deployment.py

Validates the numerical equivalence of the exported TorchScript/ONNX models
against the original PyTorch model, and measures inference latency and model sizes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from crowdflow_dna.inference.runtime import InferenceRuntime
from training.export_model import ModelExporter
from training.generate_baseline_report import generate_report_and_lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_file_size_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def measure_latency(runtime: InferenceRuntime, dummy_inputs: tuple, iterations: int = 100) -> float:
    # Warmup
    for _ in range(10):
        runtime.predict_batch(*dummy_inputs)
        
    start = time.perf_counter()
    for _ in range(iterations):
        runtime.predict_batch(*dummy_inputs)
    end = time.perf_counter()
    
    return ((end - start) * 1000.0) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (e.g., experiments/runs/.../checkpoints/best.pt)")
    parser.add_argument("--export-dir", type=str, default=None, help="Directory to save the deployment artifact. Defaults to <checkpoint_dir>/../deploy")
    args = parser.parse_args()
    
    if args.export_dir:
        export_dir = Path(args.export_dir)
    else:
        # Default to experiments/runs/<experiment>/deploy
        export_dir = Path(args.checkpoint).parent.parent / "deploy"
        
    export_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        
    logger.info("Initializing ModelExporter...")
    exporter = ModelExporter(
        config_dict=config_dict,
        checkpoint_path=args.checkpoint,
        export_dir=str(export_dir)
    )
    
    logger.info("Exporting models and validating numerical equivalence...")
    export_result = exporter.export()
    
    if not export_result.validation_success:
        logger.error(f"Numerical validation failed! Max diff: {export_result.max_output_difference}")
        sys.exit(1)
        
    pt_size = get_file_size_mb(Path(args.checkpoint))
    ts_size = get_file_size_mb(export_result.torchscript_path)
    onnx_size = get_file_size_mb(export_result.onnx_path)
    
    logger.info("Measuring InferenceRuntime latencies...")
    dummy_inputs = exporter._get_dummy_inputs(batch_size=1, seq_len=5, num_nodes=10, num_edges=20)
    
    ts_latency = 0.0
    onnx_latency = 0.0
    
    runtime = InferenceRuntime()
    if export_result.torchscript_path:
        runtime.load_model(export_result.torchscript_path)
        ts_latency = measure_latency(runtime, dummy_inputs)
        logger.info(f"TorchScript latency (batch=1): {ts_latency:.2f} ms")
        
    if export_result.onnx_path:
        try:
            runtime.load_model(export_result.onnx_path)
            onnx_latency = measure_latency(runtime, dummy_inputs)
            logger.info(f"ONNX latency (batch=1): {onnx_latency:.2f} ms")
        except Exception as e:
            logger.warning(f"Could not benchmark ONNX latency: {e}")
            
    # Generate summary report
    md = f"""# Deployment Validation Report

## Export Status
- **Method**: {export_result.export_method}
- **Validation Success**: {export_result.validation_success}
- **Max Output Difference (vs PyTorch)**: {export_result.max_output_difference:.8f}

## Model Sizes
- **PyTorch (Original)**: {pt_size:.2f} MB
- **TorchScript**: {ts_size:.2f} MB
- **ONNX**: {onnx_size:.2f} MB

## Inference Latency (Batch Size = 1)
- **TorchScript via InferenceRuntime**: {ts_latency:.2f} ms
- **ONNX via InferenceRuntime**: {onnx_latency:.2f} ms
"""
    with open(export_dir / "deployment_validation.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    logger.info(f"Deployment validation complete. Report saved to {export_dir / 'deployment_validation.md'}")

    # Generate the baseline report and lock file
    try:
        run_dir = Path(args.checkpoint).parent.parent
        dataset_dir = Path(config_dict.get("simulation", {}).get("output_dir", "data/simulated"))
        dataset_stats_json = Path("experiments/dataset_validation/dataset_statistics.json")
        
        deployment_stats = {
            "validation_success": str(export_result.validation_success),
            "max_output_difference": export_result.max_output_difference,
            "parameter_count": sum(p.numel() for p in exporter._get_dummy_inputs()[0]) * 0 + 200000, # Mock since we can't get parameter count directly here without model config loaded fully. Actually, let's just compute it from pt model if we want, or leave as placeholder. wait, the user wants parameter count!
            "pt_size": round(pt_size, 2),
            "ts_size": round(ts_size, 2),
            "onnx_size": round(onnx_size, 2),
            "ts_latency": round(ts_latency, 2),
            "onnx_latency": round(onnx_latency, 2)
        }
        
        # Load the PyTorch model to get the parameter count
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint)
        parameter_count = sum(t.numel() for t in state_dict.values())
        deployment_stats["parameter_count"] = parameter_count
        
        generate_report_and_lock(run_dir, dataset_dir, dataset_stats_json, deployment_stats)
    except Exception as e:
        logger.error(f"Failed to generate baseline report and lock file: {e}")

if __name__ == "__main__":
    main()
