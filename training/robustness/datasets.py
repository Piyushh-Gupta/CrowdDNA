"""
CrowdFlow DNA — Dataset Providers
=================================
Module: training/robustness/datasets.py

Abstracts data provision for the evaluation protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

import torch
from torch_geometric.data import Data

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset

class DatasetProvider(ABC):
    """Abstract interface for data providers supplying sequences of graphs."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[tuple[List[Data], torch.Tensor]]:
        """Yields tuples of (sequence, label)."""
        pass
        
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of sequences."""
        pass


class SequenceGraphDatasetProvider(DatasetProvider):
    """Provides sequences from a standard SequenceGraphDataset."""
    
    def __init__(self, dataset: SequenceGraphDataset) -> None:
        self.dataset = dataset
        
    def __iter__(self) -> Iterator[tuple[List[Data], torch.Tensor]]:
        # The SequenceGraphDataset's __getitem__ returns a SequenceSample
        # which has a `.graphs` attribute that is a List[Data] and `.label`.
        for i in range(len(self.dataset)):
            yield self.dataset[i].graphs, self.dataset[i].label
            
    def __len__(self) -> int:
        return len(self.dataset)
