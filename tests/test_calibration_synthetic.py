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

def test_perspective_synthetic_validation():
    # Use a genuinely perspective-distorted quadrilateral
    image_pts = [(100.0, 100.0), (900.0, 120.0), (1050.0, 900.0), (50.0, 850.0)]
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    cfg = CalibrationConfig(enabled=True, image_points=image_pts, world_points_m=world_pts)
    cfg.validate()

    calibrator = MetricCalibrator(cfg)

    import cv2
    src = np.array(image_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)

    # Let's project a known pixel. We know H transforms src to dst exactly.
    # What about the center? In world coordinates, center is (5.0, 5.0).
    # Let's project world (5, 5) back to pixel to get a guaranteed held-out test point.
    H_inv, _ = cv2.findHomography(dst, src)
    world_center = np.array([[[5.0, 5.0]]], dtype=np.float32)
    pixel_center = cv2.perspectiveTransform(world_center, H_inv)[0][0]

    # Now use the metric calibrator to map pixel_center forward to world.
    test_px = np.array([pixel_center], dtype=np.float32)
    transformed = calibrator.transform_positions(test_px)

    expected_world = np.array([5.0, 5.0], dtype=np.float32)
    error = np.linalg.norm(transformed - expected_world)
    assert error < 1e-4, f"Perspective synthetic mathematical validation failed. Error: {error}"
