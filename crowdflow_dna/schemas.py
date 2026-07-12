from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrackItem:
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]            # (cx, cy)
    velocity: Tuple[float, float]            # (vx, vy)

@dataclass
class RiskPrediction:
    region_id: int
    label: str                               # "Safe", "Congesting", "Critical"
    confidence: float