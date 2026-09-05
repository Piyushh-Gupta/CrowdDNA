import pytest
import numpy as np
from crowdflow_dna.calibration import CalibrationConfig, MetricCalibrator

def test_independent_synthetic_validation():
    # We define a known transformation mapping a 200x200 pixel square to a 10x10 meter square.
    # Image points: (0,0), (200,0), (200,200), (0,200)
    # World points: (0,0), (10,0), (10,10), (0,10)
    image_pts = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    
    cfg = CalibrationConfig(enabled=True, image_points=image_pts, world_points_m=world_pts)
    cfg.validate()
    
    calibrator = MetricCalibrator(cfg)
    
    # 5th independent point NOT used for fitting.
    # Pixel (100, 100) should map to World (5.0, 5.0) exactly.
    test_px = np.array([[100.0, 100.0]], dtype=np.float32)
    expected_world = np.array([5.0, 5.0], dtype=np.float32)
    
    transformed = calibrator.transform_positions(test_px)
    
    # Verify the held-out point maps correctly with very low numerical error.
    error = np.linalg.norm(transformed - expected_world)
    assert error < 1e-5, f"Held-out synthetic validation failed. Error: {error}"
