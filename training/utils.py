import json
import subprocess
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy arrays and scalars into JSON-serializable formats."""
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return super().default(o)


def _get_git_commit() -> str | None:
    """Returns the current git commit hash, or None if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
