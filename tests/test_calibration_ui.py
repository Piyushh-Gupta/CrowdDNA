import pytest
import numpy as np
import gradio as gr
from app import get_first_frame, handle_image_click, validate_and_preview
from crowdflow_dna.calibration_utils import draw_calibration_points, evaluate_calibration
from crowdflow_dna.calibration import CalibrationConfig
from crowdflow_dna.graph.graph_builder import GraphBuilder

def test_get_first_frame_no_video():
    img, df, msg, img2 = get_first_frame(None)
    assert img is None
    assert "No video" in msg
    
def test_handle_image_click_ordering():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    state = []
    
    # Simulate clicking
    class Evt:
        def __init__(self, idx):
            self.index = idx
            
    img1, s1, m1 = handle_image_click(Evt((10, 10)), img, state)
    assert s1 == [(10, 10)]
    
    img2, s2, m2 = handle_image_click(Evt((90, 10)), img, s1)
    img3, s3, m3 = handle_image_click(Evt((90, 90)), img, s2)
    img4, s4, m4 = handle_image_click(Evt((10, 90)), img, s3)
    
    assert len(s4) == 4
    assert s4 == [(10, 10), (90, 10), (90, 90), (10, 90)]
    assert "Now enter the real-world coordinates" in m4
    
    # 5th click should reset
    img5, s5, m5 = handle_image_click(Evt((50, 50)), img, s4)
    assert len(s5) == 1
    assert s5 == [(50, 50)]

def test_validate_and_preview_uncalibrated_disabled():
    state, msg, _, _ = validate_and_preview([], [], enable_calib=False)
    assert state == "UNVERIFIED / INCOMPLETE"
    assert "exactly 4" in msg

def test_validate_and_preview_invalid_points():
    state, msg, _, _ = validate_and_preview([(0,0)], [], enable_calib=True)
    assert state == "UNVERIFIED / INCOMPLETE"
    assert "exactly 4" in msg

def test_validate_and_preview_valid():
    state_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
    world_pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    
    state, msg, prev, _ = validate_and_preview(state_pts, world_pts, enable_calib=True)
    assert state == "CALIBRATION ENABLED"
    assert "VALID" in msg
    # Reprojection error should be extremely small
    assert "0.0000 m" in msg

def test_evaluate_calibration():
    # Synthetic mapping
    cfg = CalibrationConfig(
        enabled=True,
        image_points=[(0,0), (100,0), (100,100), (0,100)],
        world_points_m=[(0,0), (10,0), (10,10), (0,10)]
    )
    res = evaluate_calibration(cfg)
    assert res["status"] == "VALID GEOMETRY"
    assert res["fit_residual"] < 1e-5
    
    pts = res["transformed_points"]
    assert np.allclose(pts[0], [0, 0], atol=1e-5)
    assert np.allclose(pts[2], [10, 10], atol=1e-5)

def test_synthetic_graph_connectivity():
    gb = GraphBuilder(proximity_radius=2.0)
    
    # A = (1, 1), B = (2, 1), C = (5, 1)
    # A-B distance = 1m -> edge
    # A-C distance = 4m -> no edge
    pos = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [5.0, 1.0]
    ], dtype=np.float32)
    vel = np.zeros_like(pos)
    
    graph = gb.build(pos, vel)
    # Node 0 connected to 1 (both ways) -> 2 edges. Node 2 connected to nothing.
    assert graph.edge_index.shape[1] == 2
    
    # Check edges exist between 0 and 1
    edges = list(zip(graph.edge_index[0].numpy(), graph.edge_index[1].numpy()))
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (0, 2) not in edges
