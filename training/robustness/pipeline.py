"""
CrowdFlow DNA — Evaluation Pipeline
===================================
Module: training/robustness/pipeline.py

The main orchestrator for evaluation tasks.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from crowdflow_dna.inference.runtime import InferenceRuntime
from training.robustness.context import EvaluationContext
from training.robustness.protocols import EvaluationProtocol, CleanEvaluation, RobustnessEvaluation
from training.robustness.report import ReportBuilder

logger = logging.getLogger("crowddna.robustness")


# Protocol Registry
PROTOCOL_REGISTRY: Dict[str, Type[EvaluationProtocol]] = {
    "clean": CleanEvaluation,
    "robustness": RobustnessEvaluation,
    # "stress": StressEvaluation, # To be implemented
    # "regression": RegressionEvaluation, # To be implemented
}


class EvaluationPipeline:
    """Orchestrates dataset loading, protocol execution, and reporting."""
    
    def __init__(self, context: EvaluationContext) -> None:
        self.context = context
        
    def run(self) -> None:
        """Executes the pipeline."""
        logger.info(f"Starting Evaluation Pipeline. Protocol: {self.context.protocol_name}")
        
        # 1. Resolve Protocol
        protocol_cls = PROTOCOL_REGISTRY.get(self.context.protocol_name)
        if not protocol_cls:
            raise ValueError(f"Unknown protocol: {self.context.protocol_name}")
            
        protocol = protocol_cls(self.context)
        
        # 2. Setup Runtime
        logger.info(f"Loading deployment model: {self.context.deployment_model_path}")
        runtime = InferenceRuntime()
        runtime.load_model(self.context.deployment_model_path)
        
        # 3. Execute Protocol
        try:
            results = protocol.execute(runtime)
        except Exception as e:
            logger.error(f"Protocol execution failed: {e}")
            raise
            
        # 4. Generate Reports
        logger.info("Generating reports...")
        try:
            report_builder = ReportBuilder(self.context, results)
            report_builder.build()
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        logger.info("Evaluation Pipeline completed successfully.")
