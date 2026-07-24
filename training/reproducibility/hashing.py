import json
import hashlib
from typing import Dict, Any

def compute_deterministic_hash(data: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash for a dictionary."""
    # Ensure keys are sorted, and no whitespaces around separators to ensure consistency.
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    # Normalize line endings if there happen to be any strings with newlines.
    json_str = json_str.replace('\r\n', '\n')
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
