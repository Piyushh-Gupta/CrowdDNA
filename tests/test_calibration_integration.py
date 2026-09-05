import pytest
from unittest.mock import patch, MagicMock
from app import process_video
from crowdflow_dna.calibration import CalibrationConfig

@patch("crowdflow_dna.pipeline.CrowdFlowPipeline")
def test_process_video_calibration_propagation(mock_pipeline_cls):
    mock_pipeline_cls.return_value.run.return_value = MagicMock(
        output_video_path="fake.mp4",
        timeline=[],
        metadata={}
    )
    
    # Synthetic inputs from UI
    state_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
    world_coords = [[0, 0], [10, 0], [10, 10], [0, 10]]
    
    out_video, tl, md, status = process_video(
        video_file="fake_video.mp4",
        state_pts=state_pts,
        world_coords=world_coords,
        enable_calib=True
    )
    
    assert out_video == "fake.mp4"
    
    # Verify CrowdFlowPipeline was initialized with CalibrationConfig
    mock_pipeline_cls.assert_called_once()
    kwargs = mock_pipeline_cls.call_args.kwargs
    
    cfg = kwargs.get("calibration_config")
    assert cfg is not None
    assert isinstance(cfg, CalibrationConfig)
    assert cfg.enabled is True
    assert cfg.image_points == state_pts
    assert cfg.world_points_m == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
