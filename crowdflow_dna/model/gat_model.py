"""
CrowdFlow DNA — GAT Model
==========================
Module: crowdflow_dna/model/gat_model.py
Owner: Piyush Gupta (AI & Data Lead)

Implements the Graph Attention Network (GAT) encoder to classify crowd risk
from PyTorch Geometric Data objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GATConfig:
    """Configuration for CrowdDNAGAT."""
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int
    heads: int
    dropout: float

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> GATConfig:
        """Parses configuration from a dictionary (e.g., loaded from YAML)."""
        return cls(
            in_channels=int(config_dict.get("input_dim", 5)),
            hidden_channels=int(config_dict.get("gnn_hidden_dim", 64)),
            out_channels=len(config_dict.get("classes", ["Safe", "Congesting", "Critical"])),
            num_layers=int(config_dict.get("num_gnn_layers", 2)),
            heads=int(config_dict.get("gnn_heads", 4)),
            dropout=float(config_dict.get("dropout", 0.2)),
        )

    def validate(self) -> None:
        """Validates configuration parameters."""
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {self.in_channels}")
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {self.hidden_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be > 0, got {self.out_channels}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.heads < 1:
            raise ValueError(f"heads must be >= 1, got {self.heads}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


class CrowdDNAGAT(Module):
    """Graph Attention Network for crowd risk classification."""

    def __init__(self, config: GATConfig) -> None:
        """Initialises the CrowdDNAGAT model.

        Args:
            config: Configuration defining dimensions and architecture.
            
        Raises:
            ValueError: If configuration values are invalid.
        """
        super().__init__()
        config.validate()
        self.config = config

        self.drop = torch.nn.Dropout(p=config.dropout)
        self.convs = torch.nn.ModuleList()

        if config.num_layers == 1:
            # Single layer maps directly to hidden/out? Actually, if only 1 layer, 
            # we map to out_channels or hidden_channels.
            # To allow global pooling, we map to hidden_channels, then use MLP for classification.
            self.convs.append(
                GATConv(
                    config.in_channels,
                    config.hidden_channels,
                    heads=config.heads,
                    concat=False,
                    dropout=config.dropout,
                    edge_dim=4,
                )
            )
        else:
            # First layer
            self.convs.append(
                GATConv(
                    config.in_channels,
                    config.hidden_channels,
                    heads=config.heads,
                    concat=True,  # Concatenate attention heads internally
                    dropout=config.dropout,
                    edge_dim=4,
                )
            )
            # Middle layers
            for _ in range(config.num_layers - 2):
                self.convs.append(
                    GATConv(
                        config.hidden_channels * config.heads,
                        config.hidden_channels,
                        heads=config.heads,
                        concat=True,
                        dropout=config.dropout,
                        edge_dim=4,
                    )
                )
            # Last layer
            self.convs.append(
                GATConv(
                    config.hidden_channels * config.heads,
                    config.hidden_channels,
                    heads=config.heads,
                    concat=False,  # Average the heads for the final hidden state
                    dropout=config.dropout,
                    edge_dim=4,
                )
            )

        # Final MLP classifier head
        self.classifier = Linear(config.hidden_channels, config.out_channels)

    def extract_features(self, data: Data) -> Tensor:
        """Extracts graph embeddings using the GAT layers (without classifier head).

        Args:
            data: PyG Data object containing x (node features), edge_index (COO format),
                  edge_attr (edge features), and batch (node-to-graph mapping for batched inputs).

        Returns:
            Embeddings of shape (batch_size, hidden_channels).
            Zero tensor for empty graphs (0 nodes).
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Determine batch size dynamically
        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graph
        if x.size(0) == 0:
            # This guard applies primarily when CrowdDNAGAT is used standalone.
            # CrowdDNAModel validates non-empty sequences before calling extract_features.
            logger.warning("CrowdDNAGAT received an empty graph at inference time.")
            batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 1
            return torch.zeros((batch_size, self.config.hidden_channels), device=x.device)

        # Apply GAT layers with ELU activations and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.drop(x)

        # Global average pooling (N, hidden_channels) -> (Batch, hidden_channels)
        x = global_mean_pool(x, batch)
        return x

    def forward(self, data: Data) -> Tensor:
        """Forward pass of the GAT model including the classifier head.

        Args:
            data: PyG Data object containing x (node features), edge_index (COO format),
                  edge_attr (edge features), and batch (node-to-graph mapping for batched inputs).

        Returns:
            Logits of shape (batch_size, out_channels).
            Zero tensor for empty graphs (0 nodes) representing a uniform prior.
        """
        # extract_features handles the empty-graph case
        return self.classifier(self.extract_features(data))
