"""
CrowdFlow DNA — Deployment Model
=================================
Module: crowdflow_dna/model/deployment_model.py

Implements a purely tensor-native canonical deployment interface.
Reuses the GAT, Temporal Encoder, and Classifier from the training model.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import Module

from crowdflow_dna.model.crowddna_model import CrowdDNAModel, CrowdDNAModelConfig
from crowdflow_dna.model.gat_model import CrowdDNAGAT
from crowdflow_dna.model.temporal_encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class CrowdDNADeploymentModel(Module):
    """
    Tensor-native canonical deployment model.
    Accepts flat tensors representing a batch of graphs and sequence lengths,
    bypassing all PyG Data objects for TorchScript and ONNX compatibility.
    """

    def __init__(self, config: CrowdDNAModelConfig) -> None:
        """Initialises the deployment model structure.

        Args:
            config: Configuration defining dimensions and architecture.
        """
        super().__init__()
        self.config = config
        
        
        self.gat = CrowdDNAGAT(config.gat_config)
        self.temporal_encoder = TemporalEncoder(config.temporal_config)
        
        temporal_out_dim = config.temporal_config.hidden_dim
        if config.temporal_config.bidirectional:
            temporal_out_dim *= 2
            
        self.classifier = torch.nn.Linear(temporal_out_dim, config.num_classes)

    @torch.jit.unused
    def load_from_training_model(self, training_model: CrowdDNAModel) -> None:
        """Surgically injects weights from a trained model.

        Args:
            training_model: A populated CrowdDNAModel instance.
        """
        self.gat.load_state_dict(training_model.gat.state_dict())
        self.temporal_encoder.load_state_dict(training_model.temporal_encoder.state_dict())
        self.classifier.load_state_dict(training_model.classifier.state_dict())

    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_attr: Tensor, 
        batch: Tensor, 
        seq_lengths: Tensor
    ) -> Tensor:
        """
        Forward pass for inference.

        Args:
            x: Node features of shape (total_nodes, num_node_features).
            edge_index: Edges of shape (2, total_edges).
            edge_attr: Edge features of shape (total_edges, num_edge_features).
            batch: Node-to-graph mapping of shape (total_nodes,).
            seq_lengths: 1D tensor specifying the number of frames per trajectory sequence.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # 1. GAT: Tensor sequence -> Graph embeddings
        # shape: (total_frames, gat_hidden_dim)
        # We bypass extract_features(Data) to directly use pure tensors.
        graph_embeddings = self.gat.extract_features_from_tensors(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            batch=batch
        )

        # 2. Reshape into padded temporal sequences
        # We perform vectorized padding without loops or scatter to guarantee ONNX traceability
        batch_size = seq_lengths.size(0)
        max_seq_len = int(seq_lengths.max().item())
        
        # Calculate start index for each sequence in the batch
        if batch_size == 1:
            cumsum_lens = torch.zeros(1, device=seq_lengths.device, dtype=seq_lengths.dtype)
        else:
            cumsum_lens = torch.cat([
                torch.zeros(1, device=seq_lengths.device, dtype=seq_lengths.dtype), 
                seq_lengths[:-1].cumsum(0)
            ])
            
        range_tensor = torch.arange(max_seq_len, device=seq_lengths.device)
        
        # Build index grid of shape (batch_size, max_seq_len)
        indices = cumsum_lens.unsqueeze(1) + range_tensor.unsqueeze(0)
        valid_mask = range_tensor.unsqueeze(0) < seq_lengths.unsqueeze(1)
        
        # Point invalid sequence steps to a dummy zero-vector at the end of graph_embeddings
        dummy_idx = graph_embeddings.size(0)
        indices = torch.where(valid_mask, indices, torch.full_like(indices, dummy_idx))
        
        zero_vec = torch.zeros((1, graph_embeddings.size(1)), device=graph_embeddings.device, dtype=graph_embeddings.dtype)
        graph_embeddings_padded = torch.cat([graph_embeddings, zero_vec], dim=0)
        
        padded_sequences = graph_embeddings_padded[indices]

        # 3. TemporalEncoder: Graph embeddings -> Temporal representation
        # shape: (batch_size, temporal_hidden_dim)
        temporal_rep = self.temporal_encoder(padded_sequences)

        # 4. Classification head: Temporal representation -> Logits
        # shape: (batch_size, num_classes)
        logits = self.classifier(temporal_rep)

        return logits
