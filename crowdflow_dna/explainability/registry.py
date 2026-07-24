"""
Explainer Registry and Categories.

This module provides a metadata-driven registry for explanation algorithms.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Type, Callable, Any

class ExplainerCategory(Enum):
    ATTENTION_BASED = "Attention-based"
    GRADIENT_BASED = "Gradient-based"
    PERTURBATION_BASED = "Perturbation-based"
    CONFIDENCE_BASED = "Confidence-based"
    WHITE_BOX = "White-box"
    BLACK_BOX = "Black-box"
    COUNTERFACTUAL = "Counterfactual"

class RuntimeCost(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass(frozen=True)
class ExplainerMetadata:
    name: str
    category: ExplainerCategory
    supports_batch: bool
    requires_gradients: bool
    requires_hooks: bool
    deterministic: bool
    runtime_cost: RuntimeCost
    description: str

class ExplainerRegistry:
    """Registry for managing and instantiating explainers."""
    
    _registry: Dict[str, Type[Any]] = {}
    _metadata: Dict[str, ExplainerMetadata] = {}

    @classmethod
    def register(cls, metadata: ExplainerMetadata) -> Callable:
        """Decorator to register an explainer class with metadata."""
        def wrapper(explainer_cls: Type[Any]) -> Type[Any]:
            if metadata.name in cls._registry:
                raise ValueError(f"Explainer '{metadata.name}' is already registered.")
            cls._registry[metadata.name] = explainer_cls
            cls._metadata[metadata.name] = metadata
            return explainer_cls
        return wrapper

    @classmethod
    def get_explainer_class(cls, name: str) -> Type[Any]:
        """Retrieve an explainer class by name."""
        if name not in cls._registry:
            raise ValueError(f"Explainer '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def get_metadata(cls, name: str) -> ExplainerMetadata:
        """Retrieve metadata for a registered explainer."""
        if name not in cls._metadata:
            raise ValueError(f"Metadata for explainer '{name}' not found.")
        return cls._metadata[name]

    @classmethod
    def list_explainers(cls) -> Dict[str, ExplainerMetadata]:
        """List all registered explainers and their metadata."""
        return dict(cls._metadata)
