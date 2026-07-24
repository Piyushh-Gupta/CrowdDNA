"""
CrowdFlow DNA — Perturbations
=============================
Module: training/robustness/perturbations.py

Implements deterministic functional operators that mutate sequence graphs.
"""

from __future__ import annotations

import copy
from typing import List

import torch
from torch_geometric.data import Data

def drop_nodes(sequence: List[Data], p: float, generator: torch.Generator) -> List[Data]:
    """
    Randomly drops nodes (missing detections) from each frame in the sequence.
    
    Args:
        sequence: Original list of Data objects.
        p: Probability of dropping a node.
        generator: PyTorch random number generator for determinism.
        
    Returns:
        A new sequence with nodes (and their incident edges) removed.
    """
    if p <= 0.0:
        return sequence
        
    perturbed_sequence = []
    for data in sequence:
        new_data = copy.copy(data)
        num_nodes = new_data.num_nodes
        if num_nodes == 0:
            perturbed_sequence.append(new_data)
            continue
            
        # Determine which nodes to keep
        mask = torch.rand(num_nodes, generator=generator) > p
        keep_indices = mask.nonzero(as_tuple=False).view(-1)
        
        # Update node features
        new_data.x = new_data.x[keep_indices]
        new_data.num_nodes = len(keep_indices)
        
        # Update edges
        if new_data.edge_index is not None and new_data.edge_index.numel() > 0:
            # Filter edges where both src and dst are in keep_indices
            edge_mask = mask[new_data.edge_index[0]] & mask[new_data.edge_index[1]]
            new_data.edge_index = new_data.edge_index[:, edge_mask]
            if getattr(new_data, "edge_attr", None) is not None:
                new_data.edge_attr = new_data.edge_attr[edge_mask]
                
            # Remap edge indices to new node indices
            if new_data.edge_index.numel() > 0:
                mapping = torch.zeros(num_nodes, dtype=torch.long)
                mapping[keep_indices] = torch.arange(len(keep_indices), dtype=torch.long)
                new_data.edge_index = mapping[new_data.edge_index]
        
        perturbed_sequence.append(new_data)
        
    return perturbed_sequence


def add_gaussian_noise(sequence: List[Data], std: float, generator: torch.Generator) -> List[Data]:
    """
    Adds Gaussian noise to node features.
    
    Args:
        sequence: Original list of Data objects.
        std: Standard deviation of the Gaussian noise.
        generator: PyTorch random number generator for determinism.
    """
    if std <= 0.0:
        return sequence
        
    perturbed_sequence = []
    for data in sequence:
        new_data = copy.copy(data)
        if new_data.num_nodes > 0:
            noise = torch.randn(new_data.x.size(), generator=generator) * std
            new_data.x = new_data.x + noise
        perturbed_sequence.append(new_data)
        
    return perturbed_sequence


def drop_frames(sequence: List[Data], p: float, generator: torch.Generator) -> List[Data]:
    """
    Randomly replaces frames with empty frames to simulate dropped frames.
    
    Args:
        sequence: Original list of Data objects.
        p: Probability of dropping a frame.
        generator: PyTorch random number generator for determinism.
    """
    if p <= 0.0:
        return sequence
        
    perturbed_sequence = []
    for data in sequence:
        if torch.rand(1, generator=generator).item() < p:
            # Create an empty frame
            empty_data = copy.copy(data)
            empty_data.x = torch.zeros((0, data.x.size(1)), dtype=data.x.dtype)
            empty_data.num_nodes = 0
            if getattr(data, "edge_index", None) is not None:
                empty_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            if getattr(data, "edge_attr", None) is not None:
                empty_data.edge_attr = torch.zeros((0, data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
            perturbed_sequence.append(empty_data)
        else:
            perturbed_sequence.append(data)
            
    return perturbed_sequence
