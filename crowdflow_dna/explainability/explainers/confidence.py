"""
Confidence Explainer.

Analyzes raw logits to compute prediction confidence, entropy, and margin.
"""
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

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
        name="confidence",
        category=ExplainerCategory.CONFIDENCE_BASED,
        supports_batch=True,
        requires_gradients=False,
        requires_hooks=False,
        deterministic=True,
        runtime_cost=RuntimeCost.LOW,
        description="Computes prediction confidence, margin, and entropy from raw logits."
    )
)
class ConfidenceExplainer(ExplainerProtocol):
    """Explainer for prediction confidence and entropy."""
    
    def explain(
        self,
        context: ExplanationContext,
        session: ExplanationSession,
        graph_sequence: Data,
        prediction: torch.Tensor,
        sequence_id: str
    ) -> ExplanationGraph:
        """Analyzes prediction confidence."""
        
        # prediction is expected to be logits of shape (1, num_classes) if single sequence
        probs = F.softmax(prediction, dim=-1)
        
        # Calculate Entropy
        log_probs = F.log_softmax(prediction, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        
        # Calculate Margin (difference between top 1 and top 2 probabilities)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        if sorted_probs.size(-1) > 1:
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item()
        else:
            margin = 1.0
            
        confidence_analysis = {
            "entropy": entropy,
            "margin": margin,
            "max_probability": sorted_probs[0, 0].item()
        }
        
        return ExplanationGraph(
            session=session,
            explainer_name="confidence",
            sequence_id=sequence_id,
            node_importance=None,
            edge_importance=None,
            temporal_importance=None,
            confidence_analysis=confidence_analysis,
            prediction_metadata={"logits": prediction.detach().cpu().numpy().tolist()}
        )
