"""
CrowdFlow DNA — End-to-End Model
=================================
Module: crowdflow_dna/model/crowddna_model.py
Owner: Piyush Gupta (AI & Data Lead)

Implements the unified end-to-end model composing the Graph Attention Network
and Temporal Encoder into a single trainable module.

NOTE: This model currently assumes fixed-length sequences (e.g., all trajectories
in a batch have exactly 300 timesteps). It pads sequences with zeros before passing
them to the Temporal Encoder. When variable-length trajectories are introduced,
this padding logic will corrupt the GRU hidden state for shorter sequences. This
is a known limitation and will require packed sequences (e.g.,
torch.nn.utils.rnn.pack_padded_sequence) in the future.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.utils.checkpoint
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.data import Batch, Data

from crowdflow_dna.model.gat_model import CrowdDNAGAT, GATConfig
from crowdflow_dna.model.temporal_encoder import TemporalConfig, TemporalEncoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrowdDNAModelConfig:
    """Configuration for the end-to-end CrowdDNAModel."""
    gat_config: GATConfig
    temporal_config: TemporalConfig
    num_classes: int

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CrowdDNAModelConfig:
        """Parses configuration from a dictionary."""
        gat_config = GATConfig.from_dict(config_dict)
        temporal_config = TemporalConfig.from_dict(config_dict)
        classes = config_dict.get("classes", [])
        num_classes = len(classes) if classes else 3
        
        return cls(
            gat_config=gat_config,
            temporal_config=temporal_config,
            num_classes=num_classes,
        )

    def validate(self) -> None:
        """Validates configuration parameters."""
        self.gat_config.validate()
        self.temporal_config.validate()
        
        if self.gat_config.hidden_channels != self.temporal_config.input_dim:
            raise ValueError(
                f"Dimension mismatch: GAT hidden_channels ({self.gat_config.hidden_channels}) "
                f"must match TemporalEncoder input_dim ({self.temporal_config.input_dim})"
            )
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")


class CrowdDNAModel(Module):
    """
    Unified end-to-end CrowdDNA model.
    Processes sequences of graph frames through GAT and a Temporal Encoder.
    """

    def __init__(self, config: CrowdDNAModelConfig) -> None:
        """
        Initialises the CrowdDNAModel.

        Args:
            config: Configuration defining dimensions and architecture.
            
        Raises:
            ValueError: If configuration values are invalid or mismatched.
        """
        super().__init__()
        config.validate()
        self.config = config

        self.gat = CrowdDNAGAT(config.gat_config)
        self.temporal_encoder = TemporalEncoder(config.temporal_config)
        
        # Temporal encoder output dimension depends on bidirectional flag
        temporal_out_dim = config.temporal_config.hidden_dim
        if config.temporal_config.bidirectional:
            temporal_out_dim *= 2
            
        self.classifier = Linear(temporal_out_dim, config.num_classes)

    def forward(self, sequences: list[list[Data]]) -> Tensor:
        """
        Forward pass of the end-to-end model.

        Args:
            sequences: A list of sequences. Each sequence represents one trajectory 
                       sample as a list of PyG Data objects (one per frame).
                       The DataLoader collate_fn should reconstruct sequences.
                       This model does not support pre-batched inner sequences.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        if not sequences:
            raise ValueError("CrowdDNAModel received an empty batch of sequences.")
            
        # Determine sequence batching strategy and extract lengths
        seq_lengths = []
        for seq in sequences:
            if not isinstance(seq, list):
                raise TypeError(f"Expected list of list[Data], got list[{type(seq).__name__}]")
            if not seq:
                raise ValueError("Encountered an empty sequence in the batch.")
            seq_lengths.append(len(seq))
                
        # 1. GAT & 2. Reshape into padded temporal sequences
        # We process each sequence individually to prevent graph batching explosion.
        # Batching all 16x300 frames (4800 graphs, 1.9M edges) into one giant GAT pass
        # causes PyTorch to allocate massive intermediate tensors (>2GB) for attention,
        # leading to OOM during the backward pass.
        batch_size = len(sequences)
        max_seq_len = max(seq_lengths)
        embedding_dim = self.config.gat_config.hidden_channels
        
        # Use device from first graph
        device = sequences[0][0].x.device
        
        padded_sequences = torch.zeros(
            (batch_size, max_seq_len, embedding_dim), 
            device=device,
            dtype=torch.float32
        )
        
        for b_idx, seq in enumerate(sequences):
            seq_batch = Batch.from_data_list(seq)
            # Use gradient checkpointing to prevent accumulating 8GB of GAT intermediate tensors.
            # This recomputes the forward pass during backward(), saving ~7.5GB of VRAM.
            if self.training:
                # Force requires_grad=True on the input tensor to activate PyTorch's gradient
                # checkpointing engine. By default, dataset inputs have requires_grad=False,
                # which causes PyTorch to completely bypass the checkpointing mechanism.
                # This input tensor is ephemeral (not in the optimizer), so it will not cause
                # unintended parameter updates, and its gradients are freed after backward().
                seq_batch.x.requires_grad_(True)
                seq_embeddings = torch.utils.checkpoint.checkpoint(
                    self.gat.extract_features, seq_batch, use_reentrant=False
                )
            else:
                seq_embeddings = self.gat.extract_features(seq_batch)
                
            padded_sequences[b_idx, :len(seq), :] = seq_embeddings
            
        # 3. TemporalEncoder: Graph embeddings -> Temporal representation
        # shape: (batch_size, temporal_hidden_dim)
        temporal_rep = self.temporal_encoder(padded_sequences)
        
        # 4. Classification head: Temporal representation -> Logits
        # shape: (batch_size, num_classes)
        logits = self.classifier(temporal_rep)
        
        return logits
