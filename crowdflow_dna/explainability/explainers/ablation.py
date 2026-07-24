"""
Ablation Explainer.

Calculates node importance by systematically ablating (zeroing out) node features
and measuring the drop in predicted class probability.
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
        name="ablation",
        category=ExplainerCategory.PERTURBATION_BASED,
        supports_batch=False,
        requires_gradients=False,
        requires_hooks=False,
        deterministic=True,
        runtime_cost=RuntimeCost.HIGH,
        description="Measures node importance via systematic feature ablation."
    )
)
class AblationExplainer(ExplainerProtocol):
    """Explainer for node importance using ablation."""
    
    def explain(
        self,
        context: ExplanationContext,
        session: ExplanationSession,
        graph_sequence: Data,
        prediction: torch.Tensor,
        sequence_id: str
    ) -> ExplanationGraph:
        """Analyzes node importance via ablation."""
        model = context.model
        model.eval()
        
        # Base probabilities
        with torch.no_grad():
            base_probs = F.softmax(prediction, dim=-1)[0]
            pred_class = torch.argmax(base_probs).item()
            base_prob = base_probs[pred_class].item()
            
        num_nodes = graph_sequence.x.size(0)
        node_importance = torch.zeros(num_nodes, dtype=torch.float32, device=context.device)
        
        # Ablate each node (set features to zero) and measure drop in prob
        # For large graphs, this should be vectorized, but for clarity and 
        # correctness, we do it iteratively.
        for i in range(num_nodes):
            x_ablated = graph_sequence.x.clone()
            x_ablated[i] = 0.0
            
            with torch.no_grad():
                ablated_logits = model(
                    x=x_ablated,
                    edge_index=graph_sequence.edge_index,
                    edge_attr=graph_sequence.edge_attr,
                    batch=graph_sequence.batch,
                    seq_lengths=graph_sequence.seq_lengths
                )
                ablated_probs = F.softmax(ablated_logits, dim=-1)[0]
                ablated_prob = ablated_probs[pred_class].item()
                
            # Importance is the drop in probability for the predicted class
            drop = base_prob - ablated_prob
            node_importance[i] = max(0.0, drop)
            
        return ExplanationGraph(
            session=session,
            explainer_name="ablation",
            sequence_id=sequence_id,
            node_importance=node_importance,
            edge_importance=None,
            temporal_importance=None,
            confidence_analysis=None,
            prediction_metadata={"predicted_class": pred_class, "base_probability": base_prob}
        )
