from typing import Dict, Any
from training.reproducibility.metadata import ExperimentManifest

def _compare_dicts(d1: Dict[str, Any], d2: Dict[str, Any], prefix: str = "") -> list:
    """Recursively compares two dictionaries and returns a list of diff strings."""
    diffs = []
    all_keys = set(d1.keys()).union(set(d2.keys()))
    
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        
        if key not in d1:
            diffs.append(f"+ {full_key}: {d2[key]}")
        elif key not in d2:
            diffs.append(f"- {full_key}: {d1[key]}")
        else:
            val1 = d1[key]
            val2 = d2[key]
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                diffs.extend(_compare_dicts(val1, val2, full_key))
            elif val1 != val2:
                diffs.append(f"~ {full_key}: {val1} -> {val2}")
                
    return diffs

class ManifestDiffer:
    """Produces a structured human-readable diff between two manifests."""
    
    @staticmethod
    def diff_manifests(manifest1: ExperimentManifest, manifest2: ExperimentManifest) -> str:
        """Returns a formatted diff string between two manifests."""
        # Convert dataclasses to dicts for easy comparison (excluding transient fields if needed)
        import dataclasses
        d1 = dataclasses.asdict(manifest1)
        d2 = dataclasses.asdict(manifest2)
        
        diff_lines = _compare_dicts(d1, d2)
        if not diff_lines:
            return "No differences found."
            
        return "\n".join(diff_lines)
