import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class CalibrationConfig:
    enabled: bool = False
    image_points: List[Tuple[float, float]] = field(default_factory=list)
    world_points_m: List[Tuple[float, float]] = field(default_factory=list)

    def validate(self) -> None:
        if not self.enabled:
            return
        if len(self.image_points) != 4:
            raise ValueError(f"Exactly 4 image points required, got {len(self.image_points)}")
        if len(self.world_points_m) != 4:
            raise ValueError(f"Exactly 4 world points required, got {len(self.world_points_m)}")
        
        # Check collinearity via homography computation
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array(self.world_points_m, dtype=np.float32)
        
        for pts, name in [(src, 'image_points'), (dst, 'world_points_m')]:
            if not np.isfinite(pts).all():
                raise ValueError(f"{name} must contain finite numeric values.")
            
        H, status = cv2.findHomography(src, dst)
        if H is None or H.shape != (3, 3):
            raise ValueError("Failed to compute homography matrix. Points may be collinear.")

class MetricCalibrator:
    """
    Performs pixel to metric ground-plane transformation.
    
    A homography performs perspective rectification, while the destination 
    correspondences establish explicit physical metric scale (meters).
    """
    def __init__(self, config: CalibrationConfig):
        config.validate()
        self.config = config
        self.H: Optional[np.ndarray] = None
        
        if config.enabled:
            src = np.array(config.image_points, dtype=np.float32)
            dst = np.array(config.world_points_m, dtype=np.float32)
            self.H, _ = cv2.findHomography(src, dst)
            
    def transform_positions(self, positions_px: np.ndarray) -> np.ndarray:
        if not self.config.enabled or self.H is None:
            return positions_px.copy()
            
        if len(positions_px) == 0:
            return np.zeros_like(positions_px)
            
        pts = np.array([positions_px], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed[0]
