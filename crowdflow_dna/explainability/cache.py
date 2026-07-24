"""
Explanation Cache.

Provides deterministic caching of explanation results to avoid recomputing
expensive explainers like Ablation or Integrated Gradients.
"""
import hashlib
import json
import os
import pickle
from typing import Optional

from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

class CacheManager:
    """Manages reading and writing ExplanationGraph objects to disk."""

    def __init__(self, cache_dir: str = "experiments/explainability/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_hash(self, session: ExplanationSession, explainer_name: str, sequence_id: str) -> str:
        """Computes a deterministic SHA-256 hash for the cache key."""
        hash_data = {
            "deployment_model_hash": session.deployment_model_hash,
            "dataset_identifier": session.dataset_identifier,
            "configuration_hash": session.configuration_hash,
            "schema_version": session.schema_version,
            "explainer_name": explainer_name,
            "sequence_id": sequence_id
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    def get(self, session: ExplanationSession, explainer_name: str, sequence_id: str) -> Optional[ExplanationGraph]:
        """Retrieves a cached ExplanationGraph if it exists."""
        cache_key = self._compute_hash(session, explainer_name, sequence_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Cache corruption detected at {cache_path}: {e}. Removing corrupted file.")
                os.remove(cache_path)
                return None
        return None

    def set(self, graph: ExplanationGraph) -> None:
        """Stores an ExplanationGraph in the cache."""
        cache_key = self._compute_hash(graph.session, graph.explainer_name, graph.sequence_id)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)
