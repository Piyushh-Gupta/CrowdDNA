import pytest
import numpy as np
from crowdflow_dna.calibration import CalibrationConfig, MetricCalibrator

def test_invalid_point_count():
    with pytest.raises(ValueError, match="Exactly 4 image points"):
        CalibrationConfig(enabled=True, image_points=[(0,0)]).validate()

def test_invalid_world_point_count():
    with pytest.raises(ValueError, match="Exactly 4 world points"):
        CalibrationConfig(
            enabled=True, 
            image_points=[(0,0), (1,0), (1,1), (0,1)],
            world_points_m=[(0,0)]
        ).validate()

def test_collinear_points():
    cfg = CalibrationConfig(
        enabled=True,
        image_points=[(0,0), (1,0), (2,0), (3,0)],
        world_points_m=[(0,0), (1,0), (1,1), (0,1)]
    )
    with pytest.raises(ValueError, match="collinear"):
        cfg.validate()

def test_disabled_behavior():
    cfg = CalibrationConfig(enabled=False)
    cfg.validate()
    calibrator = MetricCalibrator(cfg)
    
    pts = np.array([[100, 200], [300, 400]], dtype=np.float32)
    transformed = calibrator.transform_positions(pts)
    assert np.allclose(pts, transformed)

def test_identity_mapping():
    # If image points and world points are the same, homography is identity
    pts = [(0,0), (1,0), (1,1), (0,1)]
    cfg = CalibrationConfig(enabled=True, image_points=pts, world_points_m=pts)
    calibrator = MetricCalibrator(cfg)
    
    test_pts = np.array([[0.5, 0.5], [2.0, 3.0]], dtype=np.float32)
    transformed = calibrator.transform_positions(test_pts)
    assert np.allclose(test_pts, transformed)

def test_known_scale():
    # 2x2 square in image -> 2m x 2m square in world
    cfg = CalibrationConfig(
        enabled=True,
        image_points=[(0,0), (2,0), (2,2), (0,2)],
        world_points_m=[(0,0), (2,0), (2,2), (0,2)]
    )
    calibrator = MetricCalibrator(cfg)
    transformed = calibrator.transform_positions(np.array([[1.0, 1.0]], dtype=np.float32))
    assert np.allclose(transformed, [[1.0, 1.0]])
