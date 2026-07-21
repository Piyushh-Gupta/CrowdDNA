"""
CrowdFlow DNA - Inference Runtime
=================================
Module: crowdflow_dna/inference/runtime.py

Implements a unified deployment runtime for loading exported models and performing inference.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the specified model path does not exist."""
    pass


class UnsupportedModelFormatError(Exception):
    """Raised when the model format is unknown or unsupported."""
    pass


class InferenceExecutionError(Exception):
    """Raised when model prediction fails."""
    pass


@dataclass(frozen=True)
class InferenceResult:
    """Canonical inference result from deployment models."""
    predicted_class: int
    probabilities: np.ndarray
    confidence: float
    backend: str
    inference_time_ms: float
    model_format: str
    model_version: str | None = None


class InferenceBackend(ABC):
    """Abstract interface for deployment backends."""
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Loads the model from the specified path."""
        pass

    @abstractmethod
    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        seq_lengths: Tensor
    ) -> np.ndarray:
        """Runs the forward pass and returns raw logits as a numpy array."""
        pass


class TorchScriptBackend(InferenceBackend):
    """Backend execution for TorchScript (.pt) models."""

    def __init__(self) -> None:
        self.model: torch.jit.ScriptModule | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, path: Path) -> None:
        try:
            self.model = torch.jit.load(path, map_location=self.device)
            self.model.eval()
        except Exception as e:
            raise InferenceExecutionError(f"Failed to load TorchScript model: {e}") from e

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        seq_lengths: Tensor
    ) -> np.ndarray:
        if self.model is None:
            raise InferenceExecutionError("Model not loaded.")
        
        try:
            with torch.no_grad():
                logits = self.model(
                    x.to(self.device),
                    edge_index.to(self.device),
                    edge_attr.to(self.device),
                    batch.to(self.device),
                    seq_lengths.to(self.device)
                )
                return logits.cpu().numpy()
        except Exception as e:
            raise InferenceExecutionError(f"TorchScript inference failed: {e}") from e


class ONNXBackend(InferenceBackend):
    """Backend execution for ONNX (.onnx) models."""

    def __init__(self) -> None:
        self.session: Any = None
        
    def load(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise InferenceExecutionError("onnxruntime is not installed.")
        
        try:
            # Using CPUExecutionProvider by default for cross-platform robustness.
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            raise InferenceExecutionError(f"Failed to load ONNX model: {e}") from e

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        seq_lengths: Tensor
    ) -> np.ndarray:
        if self.session is None:
            raise InferenceExecutionError("Model not loaded.")
            
        try:
            ort_inputs = {
                "x": x.cpu().numpy(),
                "edge_index": edge_index.cpu().numpy(),
                "edge_attr": edge_attr.cpu().numpy(),
                "batch": batch.cpu().numpy(),
                "seq_lengths": seq_lengths.cpu().numpy()
            }
            logits = self.session.run(None, ort_inputs)[0]
            return np.array(logits)
        except Exception as e:
            raise InferenceExecutionError(f"ONNX inference failed: {e}") from e


class InferenceRuntime:
    """
    Unified runtime for CrowdDNA inference.
    Automatically identifies model types and routes them to appropriate backends.
    """
    
    def __init__(self) -> None:
        self.backend: InferenceBackend | None = None
        self.backend_name = "unknown"
        self.model_format = "unknown"
        self.model_version: str | None = None
        
    def load_model(self, path: Path | str, version: str | None = None) -> None:
        """
        Loads the deployment model and selects the backend automatically based on extension.
        
        Args:
            path: Path to the exported model (.pt or .onnx)
            version: Optional version identifier for the model.
        """
        model_path = Path(path)
        
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")
            
        if model_path.suffix == ".pt":
            self.backend = TorchScriptBackend()
            self.backend_name = "TorchScript"
            self.model_format = "TorchScript"
        elif model_path.suffix == ".onnx":
            self.backend = ONNXBackend()
            self.backend_name = "ONNX Runtime"
            self.model_format = "ONNX"
        else:
            raise UnsupportedModelFormatError(
                f"Unsupported model format '{model_path.suffix}'. Expected .pt or .onnx."
            )
            
        self.model_version = version
        self.backend.load(model_path)
        logger.info(f"Loaded {self.model_format} model from {model_path}")

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        seq_lengths: Tensor
    ) -> InferenceResult:
        """
        Performs a single inference execution pass on the first sequence in the batch.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge features.
            batch: Node-to-graph mapping.
            seq_lengths: Sequence lengths tensor.
            
        Returns:
            InferenceResult dataclass populated with predictions and latencies.
        """
        results = self.predict_batch(x, edge_index, edge_attr, batch, seq_lengths)
        return results[0]

    def predict_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        seq_lengths: Tensor
    ) -> list[InferenceResult]:
        """
        Performs inference on a batch of sequences.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge features.
            batch: Node-to-graph mapping.
            seq_lengths: Sequence lengths tensor.
            
        Returns:
            List of InferenceResult dataclasses populated with predictions and latencies.
        """
        if self.backend is None:
            raise InferenceExecutionError("No model loaded. Call load_model() first.")
            
        start_time = time.perf_counter()
        
        logits = self.backend.predict(x, edge_index, edge_attr, batch, seq_lengths)
        
        logits_tensor = torch.from_numpy(logits)
        probabilities_batch = F.softmax(logits_tensor, dim=-1).numpy()
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        results = []
        for i in range(probabilities_batch.shape[0]):
            probs = probabilities_batch[i]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            res = InferenceResult(
                predicted_class=predicted_class,
                probabilities=probs,
                confidence=confidence,
                backend=self.backend_name,
                inference_time_ms=latency_ms,
                model_format=self.model_format,
                model_version=self.model_version
            )
            results.append(res)
            
        return results
