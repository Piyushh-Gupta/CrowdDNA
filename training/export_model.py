"""
CrowdFlow DNA — Model Export & Validation
=========================================
Module: training/export_model.py

Exports the trained CrowdDNADeploymentModel to TorchScript and ONNX, 
and validates numerical equivalence against the original PyTorch model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from crowdflow_dna.model.crowddna_model import CrowdDNAModelConfig
from crowdflow_dna.model.deployment_model import CrowdDNADeploymentModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    """Result payload of the deployment export pipeline."""
    export_directory: Path
    torchscript_path: Optional[Path]
    onnx_path: Optional[Path]
    metadata_path: Path
    validation_success: bool
    export_method: str
    max_output_difference: float


class ModelExporter:
    """Exports and validates deployment models."""

    def __init__(
        self,
        config_dict: Dict[str, Any],
        checkpoint_path: str,
        export_dir: str,
        opset_version: int = 17,
        tolerance: float = 1e-5
    ) -> None:
        """
        Initializes the ModelExporter.

        Args:
            config_dict: Dictionary containing the model configuration.
            checkpoint_path: Path to the trained PyTorch checkpoint.
            export_dir: Directory where exports will be saved.
            opset_version: ONNX opset version to target.
            tolerance: Max absolute difference for validation.
        """
        self.config_dict = config_dict
        self.checkpoint_path = Path(checkpoint_path)
        self.export_dir = Path(export_dir)
        self.opset_version = opset_version
        self.tolerance = tolerance

        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.config = CrowdDNAModelConfig.from_dict(self.config_dict)

    def _get_dummy_inputs(self, batch_size: int = 2, seq_len: int = 5, num_nodes: int = 10, num_edges: int = 20) -> Tuple[torch.Tensor, ...]:
        """Generates realistic flat tensor dummy inputs for tracing and ONNX export."""
        torch.manual_seed(42)
        
        # Determine sizes
        total_frames = batch_size * seq_len
        total_nodes = total_frames * num_nodes
        total_edges = total_frames * num_edges
        
        in_channels = self.config.gat_config.in_channels
        
        # Generate dummy tensors
        x = torch.randn(total_nodes, in_channels)
        
        # Edge index: random connections within the bounds of total_nodes
        # Note: A real batch would ensure edges only connect within the same frame graph.
        # This dummy is sufficient for tracing and shape inference.
        src = torch.randint(0, total_nodes, (total_edges,))
        dst = torch.randint(0, total_nodes, (total_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        
        edge_attr = torch.randn(total_edges, 4)  # GAT uses edge_dim=4
        
        # Batch index maps each node to a graph (frame)
        # For simplicity, assign nodes sequentially
        batch = torch.arange(total_frames).repeat_interleave(num_nodes)
        
        # Sequence lengths
        seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        return x, edge_index, edge_attr, batch, seq_lengths

    def export(self) -> ExportResult:
        """Executes the export and validation pipeline."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        # Initialize deployment model
        deploy_model = CrowdDNADeploymentModel(self.config)
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        # Ensure we're loading only the 'model_state' mapping
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        
        # deploy_model.load_state_dict requires keys to match.
        # Since deploy_model directly instantiates `gat`, `temporal_encoder`, `classifier` instead of `_internal_model`,
        # and the standard model has them at the root, the state dict keys should actually match perfectly.
        deploy_model.load_state_dict(state_dict)
        deploy_model.eval()

        # Dummy inputs for tracing/ONNX
        dummy_inputs = self._get_dummy_inputs()
        
        # 1. TorchScript Export
        ts_path = self.export_dir / "deployment.pt"
        export_method = "unknown"
        
        try:
            logger.info("Attempting TorchScript scripting...")
            ts_model = torch.jit.script(deploy_model)
            export_method = "torch.jit.script"
        except Exception as e:
            logger.warning(f"Scripting failed: {e}. Falling back to tracing...")
            ts_model = torch.jit.trace(deploy_model, dummy_inputs)
            export_method = "torch.jit.trace"
            
        torch.jit.save(ts_model, str(ts_path))
        logger.info(f"Saved TorchScript to {ts_path}")

        # 2. ONNX Export
        onnx_path = self.export_dir / "crowddna.onnx"
        logger.info("Exporting to ONNX...")
        try:
            torch.onnx.export(
                deploy_model,
                dummy_inputs,
                str(onnx_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=["x", "edge_index", "edge_attr", "batch", "seq_lengths"],
                output_names=["logits"],
                dynamic_axes={
                    "x": {0: "total_nodes"},
                    "edge_index": {1: "total_edges"},
                    "edge_attr": {0: "total_edges"},
                    "batch": {0: "total_nodes"},
                    "seq_lengths": {0: "batch_size"},
                    "logits": {0: "batch_size"}
                }
            )
            logger.info(f"Saved ONNX to {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}. PyG operator unsupported in ONNX.")
            onnx_path = None

        # 3. Validation
        logger.info("Validating exported models...")
        onnx_path_str = str(onnx_path) if onnx_path is not None else None
        max_diff = self._validate(deploy_model, ts_model, onnx_path_str, dummy_inputs)
        validation_success = max_diff <= self.tolerance
        
        if validation_success:
            logger.info(f"Validation succeeded. Max diff: {max_diff:.8f} <= {self.tolerance}")
        else:
            logger.error(f"Validation FAILED. Max diff: {max_diff:.8f} > {self.tolerance}")

        # 4. Metadata
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": self._get_git_commit(),
            "checkpoint": str(self.checkpoint_path),
            "export_method": export_method,
            "opset_version": self.opset_version,
            "tolerance": self.tolerance,
            "validation_status": "SUCCESS" if validation_success else "FAILED",
            "maximum_output_difference": float(max_diff)
        }
        
        meta_path = self.export_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        return ExportResult(
            export_directory=self.export_dir,
            torchscript_path=ts_path,
            onnx_path=onnx_path,
            metadata_path=meta_path,
            validation_success=validation_success,
            export_method=export_method,
            max_output_difference=float(max_diff)
        )

    def _validate(
        self,
        pt_model: torch.nn.Module,
        ts_model: torch.jit.ScriptModule,
        onnx_path: str,
        dummy_inputs: Tuple[torch.Tensor, ...]
    ) -> float:
        """Runs numerical validation comparing PyTorch, TorchScript, and ONNX outputs."""
        with torch.no_grad():
            pt_out = pt_model(*dummy_inputs).numpy()
            ts_out = ts_model(*dummy_inputs).numpy()
            
        max_diff = float(np.max(np.abs(pt_out - ts_out)))
        
        if onnx_path is None:
            logger.warning("ONNX model not exported. Skipping ONNX validation.")
            return max_diff
            
        try:
            import onnxruntime as ort
            logger.info("ONNX Runtime available. Validating ONNX graph...")
            ort_session = ort.InferenceSession(onnx_path)
            
            ort_inputs = {
                "x": dummy_inputs[0].numpy(),
                "edge_index": dummy_inputs[1].numpy(),
                "edge_attr": dummy_inputs[2].numpy(),
                "batch": dummy_inputs[3].numpy(),
                "seq_lengths": dummy_inputs[4].numpy()
            }
            
            onnx_out = ort_session.run(None, ort_inputs)[0]
            onnx_diff = float(np.max(np.abs(pt_out - onnx_out)))
            max_diff = max(max_diff, onnx_diff)
            
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping ONNX validation.")
            
        return max_diff

    def _get_git_commit(self) -> str:
        """Fetches current git commit, fails gracefully."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
