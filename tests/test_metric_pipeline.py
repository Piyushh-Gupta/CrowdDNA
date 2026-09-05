import pytest
import numpy as np
from crowdflow_dna.calibration import CalibrationConfig
from crowdflow_dna.pipeline import CrowdFlowPipeline
from crowdflow_dna.schemas import TrackItem

def test_pipeline_disabled_calibration():
    pipeline = CrowdFlowPipeline()
    # Mock some tracks
    tracks = [
        TrackItem(track_id=1, bbox=(0,0,100,100), centroid=(50,50), velocity=(10,10))
    ]
    pos, vel = pipeline._extract_arrays(tracks, 1000, 1000)
    assert np.allclose(pos[0], [0.05, 0.05])
    assert np.allclose(vel[0], [10, 10])

def test_pipeline_metric_calibration():
    # Setup an identity mapping just for testing velocity calculation
    pts = [(0,0), (1,0), (1,1), (0,1)]
    cfg = CalibrationConfig(enabled=True, image_points=pts, world_points_m=pts)
    
    pipeline = CrowdFlowPipeline(calibration_config=cfg)
    pipeline._dt = 0.1
    
    # First frame
    tracks1 = [
        # bottom center will be (50, 100) -> transformed to (50, 100)
        TrackItem(track_id=1, bbox=(0,0,100,100), centroid=(50,50), velocity=(10,10))
    ]
    pos1, vel1 = pipeline._extract_arrays(tracks1, 1000, 1000)
    assert np.allclose(pos1[0], [50, 100])
    assert np.allclose(vel1[0], [0, 0]) # First frame, no previous pos
    
    # Second frame, track moved
    tracks2 = [
        # bottom center will be (52, 100)
        TrackItem(track_id=1, bbox=(2,0,102,100), centroid=(52,50), velocity=(10,10))
    ]
    pos2, vel2 = pipeline._extract_arrays(tracks2, 1000, 1000)
    assert np.allclose(pos2[0], [52, 100])
    
    # velocity = dx/dt = (52-50) / 0.1 = 20
    assert np.allclose(vel2[0], [20, 0])

import pytest
import numpy as np
from crowdflow_dna.graph.graph_builder import GraphBuilder

def test_graphbuilder_metric_distance():
    # In metric calibration mode, proximity_radius is set to 2.0
    gb = GraphBuilder(proximity_radius=2.0)
    
    # 2 agents, 1 meter apart -> should create edges
    pos1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    vel1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    graph1 = gb.build(pos1, vel1)
    
    # Directed edges for (0->1) and (1->0)
    assert graph1.edge_index.shape[1] == 2
    
    # 2 agents, 3 meters apart -> should NOT create edges
    pos2 = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    vel2 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    graph2 = gb.build(pos2, vel2)
    
    assert graph2.edge_index.shape[1] == 0
