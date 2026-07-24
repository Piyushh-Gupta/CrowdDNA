from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TrackItem:
    """Represents a tracked pedestrian in a single video frame."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]            # (cx, cy)
    velocity: Tuple[float, float]            # (vx, vy)

@dataclass(frozen=True)
class RiskPrediction:
    """Represents a risk prediction output by the deployment pipeline."""
    region_id: int
    label: str                               # "Safe", "Congesting", "Critical"
    confidence: float