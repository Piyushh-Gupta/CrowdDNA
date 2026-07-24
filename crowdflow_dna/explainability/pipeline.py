"""
Explanation Pipeline.

Coordinates datasets, runs InferenceRuntime, passes results to the ExplanationEngine,
and invokes reporters and plotters.
"""
import os
import uuid
import time
import logging
import torch

from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.engine import ExplanationEngine
from crowdflow_dna.explainability.reporting.registry import ReporterRegistry
from crowdflow_dna.explainability.plotting.registry import PlottingRegistry

logger = logging.getLogger(__name__)

class ExplanationPipeline:
    """Orchestrates the explainability workflow."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.engine = ExplanationEngine()
        
    def _create_session(self, context: ExplanationContext, duration: float) -> ExplanationSession:
        return ExplanationSession(
            session_id=str(uuid.uuid4()),
            crowddna_version="1.0.0",
            schema_version="1.0",
            deployment_model_hash="unknown", # In practice, compute hash of model file
            deployment_artifact_hash="unknown", 
            dataset_identifier="unknown", # Replace with dataset hash/id
            enabled_explainers=tuple(context.configuration.get("explainers", [])),
            configuration_hash="unknown", 
            execution_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            execution_duration=duration,
            output_directory=context.output_directory,
            cache_status="enabled"
        )
        
    def run(self, context: ExplanationContext):
        """Executes the pipeline."""
        start_time = time.time()
        
        # We would iterate over the dataset and run inference
        # Since we are mocking heavy inference, we assume context.dataset_provider
        # can be iterated. For this implementation, we will just grab the first batch
        # if possible, or simulate it.
        
        graphs = []
        dataset = context.dataset_provider
        
        # Here we need to run context.runtime
        # We simulate the pipeline running on one graph sequence for testing
        
        # Note: the real pipeline would do:
        # for batch in dataset:
        #    prediction = context.runtime.forward(batch)
        #    batch_graphs = self.engine.explain(context, session, batch, prediction)
        #    graphs.extend(batch_graphs)
        
        # Because we don't want to actually run the full inference loop here,
        # we assume dataset yields (sequence_id, data) tuples
        
        duration = time.time() - start_time
        session = self._create_session(context, duration)
        
        for sequence_id, data in dataset:
            # We mock inference for testing
            # Actually, context.runtime is not mocked, we must call it:
            data = data.to(context.device)
            # Forward pass
            with torch.no_grad():
                logits = context.model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch,
                    seq_lengths=data.seq_lengths
                )
            
            sequence_graphs = self.engine.explain(
                context=context,
                session=session,
                graph_sequence=data,
                prediction=logits,
                sequence_id=sequence_id
            )
            graphs.extend(sequence_graphs)
            
        logger.info("Generating reports...")
        for reporter_name in ["markdown", "json", "csv"]:
            reporter_cls = ReporterRegistry.get_reporter(reporter_name)
            reporter = reporter_cls()
            reporter.generate(session, graphs)
            
        logger.info("Generating plots...")
        for plotter_cls in PlottingRegistry.get_plotters():
            plotter = plotter_cls()
            plotter.plot(session, graphs)
            
        logger.info("Explainability pipeline completed.")
