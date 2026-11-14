import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from crowdflow_dna.geometry_validation import _check_polygon_validity

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
        
        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array(self.world_points_m, dtype=np.float32)
        
        for pts, name in [(src, 'image_points'), (dst, 'world_points_m')]:
            if not np.isfinite(pts).all():
                raise ValueError(f"{name} must contain finite numeric values.")
            
        # Robust geometric validation
        _check_polygon_validity(src, 'image_points', min_area=1.0) # Image area should be > 10 pixels
        _check_polygon_validity(dst, 'world_points_m', min_area=1e-3) # World area should be non-negligible
            
        H, status = cv2.findHomography(src, dst)
        if H is None or H.shape != (3, 3) or not np.isfinite(H).all():
            raise ValueError("Failed to compute a valid, finite homography matrix. Points may be collinear or ill-conditioned.")

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
        
        if not np.isfinite(transformed).all():
            raise ValueError("Transformation resulted in non-finite coordinates. The homography may be numerically unstable.")
            
        return transformed[0]
