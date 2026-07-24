import random
from abc import ABC, abstractmethod

class SamplingPolicy(ABC):
    """Abstract base class for sampling policies."""
    @abstractmethod
    def should_sample(self) -> bool:
        pass

class AlwaysSample(SamplingPolicy):
    """Samples every single observation."""
    def should_sample(self) -> bool:
        return True

class EveryNSample(SamplingPolicy):
    """Samples exactly every Nth observation."""
    def __init__(self, n: int):
        self.n = max(1, n)
        self._count = 0
        
    def should_sample(self) -> bool:
        self._count += 1
        if self._count >= self.n:
            self._count = 0
            return True
        return False

class ProbabilitySample(SamplingPolicy):
    """Samples randomly based on a probability threshold (0.0 to 1.0)."""
    def __init__(self, probability: float):
        self.probability = max(0.0, min(1.0, probability))
        
    def should_sample(self) -> bool:
        return random.random() < self.probability

class AdaptiveSample(SamplingPolicy):
    """Dynamically adjusts sampling rate based on system conditions. (Placeholder logic)"""
    def __init__(self, initial_rate: float = 1.0):
        self.current_rate = initial_rate
        
    def should_sample(self) -> bool:
        # In a real system, current_rate might be adjusted externally via a feedback loop.
        return random.random() < self.current_rate
