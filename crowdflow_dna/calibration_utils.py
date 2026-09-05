import cv2
import numpy as np
from typing import Dict, Any, Tuple
from crowdflow_dna.calibration import MetricCalibrator, CalibrationConfig

def evaluate_calibration(config: CalibrationConfig) -> Dict[str, Any]:
    config.validate()
    calibrator = MetricCalibrator(config)
    
    # Calculate fit residual (not independent accuracy!)
    src = np.array(config.image_points, dtype=np.float32)
    dst = np.array(config.world_points_m, dtype=np.float32)
    
    transformed_src = calibrator.transform_positions(src)
    
    # Residual error is the Euclidean distance between predicted and actual dst points for the SAME fitting points
    errors = np.linalg.norm(transformed_src - dst, axis=1)
    fit_residual = float(np.mean(errors))
    
    return {
        "fit_residual": fit_residual,
        "transformed_points": transformed_src.tolist(),
        "status": "VALID GEOMETRY",
        "homography": calibrator.H.tolist() if calibrator.H is not None else None
    }

def draw_calibration_points(image: np.ndarray, points: list) -> np.ndarray:
    """Draw points and connecting polygon on the image."""
    out_img = image.copy()
    if not points:
        return out_img
        
    pts = np.array(points, np.int32)
    
    if len(pts) > 1:
        is_closed = (len(pts) == 4)
        cv2.polylines(out_img, [pts], is_closed, (0, 255, 0), 2)
        
    for i, pt in enumerate(pts):
        cv2.circle(out_img, tuple(pt), 5, (0, 0, 255), -1)
        cv2.putText(out_img, str(i+1), (pt[0]+10, pt[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
    return out_img
