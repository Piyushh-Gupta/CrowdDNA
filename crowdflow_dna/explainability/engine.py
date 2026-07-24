"""
Explanation Engine.

Responsible for executing explainers, managing hooks, collecting intermediate
activations, and producing the ExplanationGraph.
"""
import logging
from typing import List
import torch
from torch_geometric.data import Data

from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph
from crowdflow_dna.explainability.registry import ExplainerRegistry
from crowdflow_dna.explainability.cache import CacheManager

logger = logging.getLogger(__name__)

class ExplanationEngine:
    """Executes explainers and produces ExplanationGraphs."""
    
    def __init__(self, cache_dir: str = "experiments/explainability/cache"):
        self.cache = CacheManager(cache_dir=cache_dir)
        
    def explain(
        self,
        context: ExplanationContext,
        session: ExplanationSession,
        graph_sequence: Data,
        prediction: torch.Tensor,
        sequence_id: str
    ) -> List[ExplanationGraph]:
        """
        Runs all enabled explainers for a given sequence and prediction.
        """
        graphs = []
        
        for explainer_name in session.enabled_explainers:
            try:
                # metadata = ExplainerRegistry.get_metadata(explainer_name)
                
                # Check cache first
                cached_graph = self.cache.get(session, explainer_name, sequence_id)
                if cached_graph is not None:
                    logger.info(f"Cache hit for {explainer_name} on sequence {sequence_id}")
                    graphs.append(cached_graph)
                    continue
                
                # Execute explainer
                logger.info(f"Executing {explainer_name} on sequence {sequence_id}")
                explainer_cls = ExplainerRegistry.get_explainer_class(explainer_name)
                explainer = explainer_cls()
                
                graph = explainer.explain(
                    context=context,
                    session=session,
                    graph_sequence=graph_sequence,
                    prediction=prediction,
                    sequence_id=sequence_id
                )
                
                # Store in cache
                self.cache.set(graph)
                graphs.append(graph)
                
            except Exception as e:
                logger.error(f"Failed to execute explainer {explainer_name}: {e}")
                
        return graphs
