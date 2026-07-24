import json
import dataclasses
import os

from training.reproducibility.metadata import ExperimentManifest, EnvironmentSnapshot
from training.reproducibility.migration import SchemaMigrator

class ManifestManager:
    """Manages the reading and writing of ExperimentManifest JSON files."""
    
    @staticmethod
    def save(manifest: ExperimentManifest, output_path: str):
        """Serializes the manifest to a JSON file."""
        data = dataclasses.asdict(manifest)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)
            
    @staticmethod
    def load(input_path: str) -> ExperimentManifest:
        """Loads and deserializes a manifest from a JSON file, applying migrations."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Manifest not found at {input_path}")
            
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
            
        migrated_data = SchemaMigrator.migrate(raw_data)
        
        # Deserialize to dataclasses
        env_data = migrated_data.pop("environment")
        env_snapshot = EnvironmentSnapshot(
            **{k: v for k, v in env_data.items() if k != "pip_freeze"},
            pip_freeze=tuple(env_data["pip_freeze"])
        )
        
        migrated_data["environment"] = env_snapshot
        migrated_data["lineage_parents"] = tuple(migrated_data["lineage_parents"])
        
        return ExperimentManifest(**migrated_data)
