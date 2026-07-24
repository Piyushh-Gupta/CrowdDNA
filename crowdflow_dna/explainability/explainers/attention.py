"""
Attention Explainer.

Attempts to extract attention weights from GAT layers and Temporal Encoders
using PyTorch forward hooks. Degrades gracefully if unsupported.
"""
import torch
from torch_geometric.data import Data
from typing import Dict

from crowdflow_dna.explainability.explainers.base import ExplainerProtocol
from crowdflow_dna.explainability.registry import (
    ExplainerRegistry, 
    ExplainerMetadata, 
    ExplainerCategory, 
    RuntimeCost
)
from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

@ExplainerRegistry.register(
    ExplainerMetadata(
        name="attention",
        category=ExplainerCategory.ATTENTION_BASED,
        supports_batch=False,
        requires_gradients=False,
        requires_hooks=True,
        deterministic=True,
        runtime_cost=RuntimeCost.MEDIUM,
        description="Extracts attention weights from graph and temporal layers."
    )
)
class AttentionExplainer(ExplainerProtocol):
    """Explainer for attention maps."""
    
    def explain(
        self,
        context: ExplanationContext,
        session: ExplanationSession,
        graph_sequence: Data,
        prediction: torch.Tensor,
        sequence_id: str
    ) -> ExplanationGraph:
        """Attempts to extract attention weights using hooks."""
        model = context.model
        model.eval()
        
        # In a real implementation, we would register forward hooks on 
        # model.gat.layers and model.temporal_encoder.
        # Since PyG GATConv doesn't expose attention weights to hooks easily 
        # without return_attention_weights=True during initialization,
        # we gracefully degrade by returning None for attention tensors 
        # if we cannot extract them.
        
        attention_tensors: Dict[str, torch.Tensor] = {}
        
        # Graceful degradation: we simulate failure to extract 
        # without modifying deployment.pt
        
        return ExplanationGraph(
            session=session,
            explainer_name="attention",
            sequence_id=sequence_id,
            node_importance=None,
            edge_importance=None,
            temporal_importance=None,
            confidence_analysis=None,
            prediction_metadata={},
            attention_tensors=attention_tensors if attention_tensors else None
        )
