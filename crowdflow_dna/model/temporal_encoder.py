"""
CrowdFlow DNA — Temporal Encoder
=================================
Module: crowdflow_dna/model/temporal_encoder.py
Owner: Piyush Gupta (AI & Data Lead)

Implements the Temporal Sequence Encoder using a GRU to process temporal 
sequences of graph embeddings produced by the GAT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import GRU, Module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for the TemporalEncoder."""
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    bidirectional: bool

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> TemporalConfig:
        """Parses configuration from a dictionary."""
        return cls(
            input_dim=int(config_dict.get("gnn_hidden_dim", 64)),
            hidden_dim=int(config_dict.get("gru_hidden_dim", 64)),
            num_layers=int(config_dict.get("gru_num_layers", 1)),
            dropout=float(config_dict.get("gru_dropout", 0.0)),
            bidirectional=bool(config_dict.get("gru_bidirectional", False)),
        )

    def validate(self) -> None:
        """Validates configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


class TemporalEncoder(Module):
    """
    Temporal sequence encoder using a Gated Recurrent Unit (GRU).
    Processes sequences of graph embeddings into a single temporal representation.
    """

    def __init__(self, config: TemporalConfig) -> None:
        """
        Initialises the TemporalEncoder.

        Args:
            config: Configuration defining dimensions and architecture.
            
        Raises:
            ValueError: If configuration values are invalid.
        """
        super().__init__()
        config.validate()
        self.config = config
        self.input_dim = config.input_dim
        self.bidirectional = config.bidirectional

        if config.num_layers == 1 and config.dropout > 0.0:
            logger.warning("Dropout is > 0 but num_layers is 1. PyTorch GRU ignores dropout for a single layer.")

        self.gru = GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TemporalEncoder.

        Args:
            x: Sequence tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            The final hidden representation of shape (batch_size, temporal_hidden_dim),
            where temporal_hidden_dim is hidden_dim (or hidden_dim * 2 if bidirectional).
        """
        if not isinstance(x, Tensor):
            raise ValueError("Input must be a torch.Tensor")
            
        if x.ndim != 3:
            raise ValueError(f"TemporalEncoder expects input of shape (batch, seq_len, input_dim), got {x.shape}")
            
        if x.size(0) < 1:
            raise ValueError(f"Batch size must be >= 1, got {x.size(0)}")
            
        if x.size(1) == 0:
            raise ValueError("TemporalEncoder received a sequence of length 0. Filter empty windows before calling forward().")
            
        if x.size(2) != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {x.size(2)}")
            
        # GRU outputs:
        # out: (batch_size, seq_len, num_directions * hidden_size)
        # hn: (num_layers * num_directions, batch_size, hidden_size)
        _, hn = self.gru(x)

        if self.bidirectional:
            # For bidirectional GRU, hn contains forward and backward states interleaved by layer.
            # The final layer's forward state is hn[-2] and backward state is hn[-1].
            hidden_forward = hn[-2]
            hidden_backward = hn[-1]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=-1)
        else:
            # For unidirectional GRU, the final layer's state is simply the last element.
            final_hidden = hn[-1]

        return final_hidden
