"""
CrowdFlow DNA — Evaluation Protocols
====================================
Module: training/robustness/protocols.py

Defines standard evaluation procedures independently of the execution pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch

from crowdflow_dna.inference.runtime import InferenceRuntime
from crowdflow_dna.inference.sequence_buffer import SequenceBuffer
from training.robustness.context import EvaluationContext
from training.robustness.metrics import METRIC_REGISTRY, Metric
from training.robustness.scenarios import SCENARIO_REGISTRY

logger = logging.getLogger("crowddna.robustness")


class EvaluationProtocol(ABC):
    """Abstract interface for a specific testing protocol."""
    
    def __init__(self, context: EvaluationContext) -> None:
        self.context = context
        
        # Instantiate active metrics
        self.metrics: List[Metric] = [
            metric_cls() for metric_cls in METRIC_REGISTRY.values()
        ]
        
    def _run_inference(self, runtime: InferenceRuntime, scenario=None) -> Dict[str, float]:
        """Helper to run inference on the entire dataset and collect metrics."""
        
        # Set up a random generator with the master seed, seeded uniquely per scenario
        seed = self.context.random_seed
        if scenario:
            seed += hash(scenario.name) % 1000000
        generator = torch.Generator().manual_seed(seed)
        
        for sequence, label in self.context.dataset_provider:
            if scenario:
                perturbed_sequence = scenario.apply(sequence, generator)
            else:
                perturbed_sequence = sequence
                
            buffer = SequenceBuffer(window_size=len(perturbed_sequence))
            for frame in perturbed_sequence:
                buffer.push(frame)
                
            if buffer.is_ready:
                batch = buffer.assemble()
                
                import time
                start = time.perf_counter()
                try:
                    result = runtime.predict(
                        x=batch.x.to(self.context.device),
                        edge_index=batch.edge_index.to(self.context.device),
                        edge_attr=batch.edge_attr.to(self.context.device),
                        batch=batch.batch.to(self.context.device),
                        seq_lengths=batch.seq_lengths.to(self.context.device)
                    )
                except Exception as e:
                    logger.warning(f"Inference failed on sequence: {e}")
                    # Log failure and skip
                    continue
                    
                inf_time = (time.perf_counter() - start) * 1000.0
                
                # Batch size is 1. We just reshape to (1, num_classes)
                probs = result.probabilities
                if probs.ndim == 1:
                    probs = probs[np.newaxis, :]
                    
                preds = np.array([result.predicted_class])
                
                targets = label.unsqueeze(0).cpu().numpy() # Shape (1,)
                
                for metric in self.metrics:
                    metric.update(preds, probs, targets, inf_time)
                    
        return {metric.get_name(): metric.compute() for metric in self.metrics}

    @abstractmethod
    def execute(self, runtime: InferenceRuntime) -> Dict[str, Any]:
        """Executes the protocol and returns results."""
        pass


class CleanEvaluation(EvaluationProtocol):
    """Evaluates the model without any perturbations."""
    
    def execute(self, runtime: InferenceRuntime) -> Dict[str, Any]:
        logger.info("Executing CleanEvaluation...")
        for metric in self.metrics:
            metric.reset()
            
        results = self._run_inference(runtime)
        return {"clean": results}


class RobustnessEvaluation(EvaluationProtocol):
    """Evaluates the model across all registered scenarios."""
    
    def execute(self, runtime: InferenceRuntime) -> Dict[str, Any]:
        logger.info("Executing RobustnessEvaluation...")
        
        all_results = {}
        
        # 1. Base clean evaluation
        for metric in self.metrics:
            metric.reset()
        all_results["baseline"] = self._run_inference(runtime)
        
        # 2. Iterate over scenarios
        for name, scenario_cls in SCENARIO_REGISTRY.items():
            logger.info(f"Running scenario: {name}")
            for metric in self.metrics:
                metric.reset()
                
            # Parameterize scenario (could be driven by config)
            params = self.context.config.get("scenarios", {}).get(name, {})
            scenario = scenario_cls(name, params)
            
            all_results[name] = self._run_inference(runtime, scenario=scenario)
            
        return all_results


class StressEvaluation(EvaluationProtocol):
    """
    Evaluates the model under extreme conditions designed to force failure.
    
    Status: DEFERRED. This is a planned extension and is not yet implemented.
    """
    
    def execute(self, runtime: InferenceRuntime) -> Dict[str, Any]:
        raise NotImplementedError("StressEvaluation is a planned extension and not yet implemented.")


class RegressionEvaluation(EvaluationProtocol):
    """
    Evaluates the model specifically against a historical database of failure cases.
    
    Status: DEFERRED. This is a planned extension and is not yet implemented.
    """
    
    def execute(self, runtime: InferenceRuntime) -> Dict[str, Any]:
        raise NotImplementedError("RegressionEvaluation is a planned extension and not yet implemented.")
