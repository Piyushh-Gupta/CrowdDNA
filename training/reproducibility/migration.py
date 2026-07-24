from typing import Dict, Any

class SchemaMigrator:
    """Migrates older reproducibility manifests to the current schema version."""
    
    CURRENT_VERSION = "1.0"
    
    @classmethod
    def migrate(cls, raw_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Applies necessary structural transformations to upgrade the manifest."""
        version = raw_manifest.get("manifest_version", "0.9")
        
        if version == cls.CURRENT_VERSION:
            return raw_manifest
            
        # Example migration path
        if version == "0.9":
            raw_manifest = cls._migrate_09_to_10(raw_manifest)
            
        return raw_manifest

    @classmethod
    def _migrate_09_to_10(cls, raw_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Migration logic from 0.9 to 1.0."""
        # 1.0 introduced crowddna_schema and validator_metadata
        if "crowddna_schema" not in raw_manifest:
            raw_manifest["crowddna_schema"] = "v1.0.0"
        if "validator_metadata" not in raw_manifest:
            raw_manifest["validator_metadata"] = {}
            
        raw_manifest["manifest_version"] = "1.0"
        return raw_manifest
