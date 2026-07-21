"""Unit tests for CrowdFlowPipeline (crowdflow_dna/pipeline.py).

All six component dependencies are mocked — no real video, no YOLO weights,
no GPU, no torch_geometric required. Tests cover orchestration logic,
data routing, dummy-mode behaviour, model injection, error propagation,
and the _extract_arrays() helper.
"""

import sys
from dataclasses import fields
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna.errors import CrowdFlowError, ModelInferenceError, VideoCorruptionError
from crowdflow_dna.pipeline import CrowdFlowPipeline, PipelineResult
from crowdflow_dna.rendering.timeline import TimelineEntry
from crowdflow_dna.schemas import RiskPrediction, TrackItem

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_W, _H = 640, 480  # synthetic frame dimensions


def _make_frame() -> np.ndarray:
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def _make_track(track_id: int = 1, cx: float = 100.0, cy: float = 100.0) -> TrackItem:
    return TrackItem(
        track_id=track_id,
        bbox=(cx - 25, cy - 50, cx + 25, cy + 50),
        centroid=(cx, cy),
        velocity=(2.0, 0.0),
    )


def _make_pred(region_id: int = 1, label: str = "Safe", conf: float = 0.9) -> RiskPrediction:
    return RiskPrediction(region_id=region_id, label=label, confidence=conf)


def _make_metadata(n_frames: int = 3) -> Dict[str, Any]:
    return {
        "fps": 30.0,
        "width": _W,
        "height": _H,
        "frame_count": n_frames * 5,
        "duration_seconds": float(n_frames * 5) / 30.0,
        "sample_rate": 5,
    }


# Patch targets (all relative to where the names are used in pipeline.py)
_PATCH_INGESTOR = "crowdflow_dna.pipeline.VideoIngestor"
_PATCH_DETECTOR = "crowdflow_dna.pipeline.Yolov8Detector"
_PATCH_TRACKER  = "crowdflow_dna.pipeline.ByteTracker"
_PATCH_ANNOTATOR = "crowdflow_dna.pipeline.FrameAnnotator"
_PATCH_TIMELINE = "crowdflow_dna.pipeline.TimelineBuilder"
_PATCH_GRAPH    = "crowdflow_dna.pipeline.GraphBuilder"


@contextmanager
def _make_pipeline_mocks(
    n_frames: int = 3,
    tracks_per_frame: int = 1,
    detections_per_frame: int = 1,
):
    """Return a tuple of (pipeline, mock_dict) with all deps mocked."""
    # Import inside test so that mocks apply at module-load time
    from crowdflow_dna.pipeline import CrowdFlowPipeline

    frames = [_make_frame() for _ in range(n_frames)]
    metadata = _make_metadata(n_frames)
    tracks = [_make_track(i + 1) for i in range(tracks_per_frame)]
    detections = [(0.0, 0.0, 50.0, 50.0, 0.9, 0)] * detections_per_frame

    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH)    as mock_gb,
    ):
        mock_ing.return_value.load.return_value = (frames, metadata)
        mock_det.return_value.detect.return_value = detections
        mock_trk.return_value.update.return_value = tracks
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = [
            TimelineEntry(frame_index=i, predictions=[]) for i in range(n_frames)
        ]
        mock_tl.return_value.reset.return_value = None
        mock_gb.return_value.build.return_value = MagicMock()

        pipeline = CrowdFlowPipeline()
        yield pipeline, {
            "ingestor": mock_ing.return_value,
            "detector": mock_det.return_value,
            "tracker":  mock_trk.return_value,
            "annotator": mock_ann.return_value,
            "timeline": mock_tl.return_value,
            "graph":    mock_gb.return_value,
            "frames":   frames,
            "metadata": metadata,
            "tracks":   tracks,
        }


# ---------------------------------------------------------------------------
# PipelineResult dataclass contract
# ---------------------------------------------------------------------------


def test_pipeline_result_has_correct_fields() -> None:
    """PipelineResult must have annotated_frames, timeline, metadata."""
    field_names = {f.name for f in fields(PipelineResult)}
    assert field_names == {"annotated_frames", "timeline", "metadata"}


def test_pipeline_result_defaults_to_empty() -> None:
    """PipelineResult with no args must have empty lists and dict."""
    r = PipelineResult()
    assert r.annotated_frames == []
    assert r.timeline == []
    assert r.metadata == {}


# ---------------------------------------------------------------------------
# run() — return type and basic structure
# ---------------------------------------------------------------------------


def test_run_returns_pipeline_result() -> None:
    """run() must return a PipelineResult instance."""
    with _make_pipeline_mocks(n_frames=2) as (pipeline, mocks):
        result = pipeline.run("fake.mp4")
    assert isinstance(result, PipelineResult)


def test_annotated_frames_length_matches_sampled_frames() -> None:
    """One annotated frame must be produced per sampled frame."""
    n = 4
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        result = pipeline.run("fake.mp4")
    assert len(result.annotated_frames) == n


def test_timeline_length_matches_sampled_frames() -> None:
    """One TimelineEntry must be recorded per sampled frame."""
    n = 3
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        result = pipeline.run("fake.mp4")
    assert len(result.timeline) == n


def test_metadata_passed_through_to_result() -> None:
    """PipelineResult.metadata must equal the ingestion metadata dict."""
    with _make_pipeline_mocks(n_frames=2) as (pipeline, mocks):
        result = pipeline.run("fake.mp4")
    assert result.metadata == mocks["metadata"]


def test_annotated_frames_are_ndarrays() -> None:
    """Every annotated frame must be a np.ndarray."""
    with _make_pipeline_mocks(n_frames=2) as (pipeline, mocks):
        result = pipeline.run("fake.mp4")
    for f in result.annotated_frames:
        assert isinstance(f, np.ndarray)


# ---------------------------------------------------------------------------
# run() — component call counts
# ---------------------------------------------------------------------------


def test_detector_called_once_per_frame() -> None:
    """detect() must be called exactly N times for N frames."""
    n = 3
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        pipeline.run("fake.mp4")
    assert mocks["detector"].detect.call_count == n


def test_tracker_called_once_per_frame() -> None:
    """update() must be called exactly N times for N frames."""
    n = 3
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        pipeline.run("fake.mp4")
    assert mocks["tracker"].update.call_count == n


def test_annotator_called_once_per_frame() -> None:
    """annotate() must be called exactly N times for N frames."""
    n = 3
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        pipeline.run("fake.mp4")
    assert mocks["annotator"].annotate.call_count == n


def test_timeline_record_called_once_per_frame() -> None:
    """timeline.record() must be called exactly N times for N frames."""
    n = 3
    with _make_pipeline_mocks(n_frames=n) as (pipeline, mocks):
        pipeline.run("fake.mp4")
    assert mocks["timeline"].record.call_count == n


def test_timeline_reset_called_on_each_run() -> None:
    """timeline.reset() must be called at the start of every run()."""
    with _make_pipeline_mocks(n_frames=2) as (pipeline, mocks):
        pipeline.run("fake.mp4")
        pipeline.run("fake.mp4")
    assert mocks["timeline"].reset.call_count == 2


# ---------------------------------------------------------------------------
# run() — dummy mode (model_fn=None)
# ---------------------------------------------------------------------------


def test_no_model_fn_runs_without_error() -> None:
    """Pipeline with model_fn=None must complete without raising."""
    with _make_pipeline_mocks(n_frames=2) as (pipeline, mocks):
        try:
            pipeline.run("fake.mp4")
        except Exception as exc:
            pytest.fail(f"Pipeline raised in dummy mode: {exc}")


def test_no_model_fn_graph_not_built() -> None:
    """GraphBuilder.build() must NOT be called when model_fn is None."""
    with _make_pipeline_mocks(n_frames=2, tracks_per_frame=1) as (pipeline, mocks):
        pipeline.run("fake.mp4")
    mocks["graph"].build.assert_not_called()


# ---------------------------------------------------------------------------
# run() — model_fn injection
# ---------------------------------------------------------------------------


def test_model_fn_invoked_when_provided() -> None:
    """model_fn must be called when tracks are present."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    mock_model = MagicMock(return_value=[_make_pred()])
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH)    as mock_gb,
    ):
        frames = [_make_frame(), _make_frame()]
        metadata = _make_metadata(2)
        mock_ing.return_value.load.return_value = (frames, metadata)
        mock_det.return_value.detect.return_value = [(0.0, 0.0, 50.0, 50.0, 0.9, 0)]
        mock_trk.return_value.update.return_value = [_make_track()]
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = []
        mock_gb.return_value.build.return_value = MagicMock()

        pipeline = CrowdFlowPipeline(model_fn=mock_model)
        pipeline.run("fake.mp4")

    assert mock_model.call_count == 2  # once per frame


def test_model_fn_not_called_when_no_tracks() -> None:
    """model_fn must NOT be called when the tracker returns no tracks."""
    mock_model = MagicMock(return_value=[])
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH),
    ):
        frames = [_make_frame()]
        metadata = _make_metadata(1)
        mock_ing.return_value.load.return_value = (frames, metadata)
        mock_det.return_value.detect.return_value = []
        mock_trk.return_value.update.return_value = []   # ← no tracks
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = []

        pipeline = CrowdFlowPipeline(model_fn=mock_model)
        pipeline.run("fake.mp4")

    mock_model.assert_not_called()


def test_model_predictions_passed_to_annotator() -> None:
    """Predictions returned by model_fn must be forwarded to annotate()."""
    pred = _make_pred(1, "Critical", 0.95)
    mock_model = MagicMock(return_value=[pred])
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH)    as mock_gb,
    ):
        frames = [_make_frame()]
        metadata = _make_metadata(1)
        mock_ing.return_value.load.return_value = (frames, metadata)
        mock_det.return_value.detect.return_value = [(0.0, 0.0, 50.0, 50.0, 0.9, 0)]
        mock_trk.return_value.update.return_value = [_make_track()]
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = []
        mock_gb.return_value.build.return_value = MagicMock()

        pipeline = CrowdFlowPipeline(model_fn=mock_model)
        pipeline.run("fake.mp4")

    _, call_kwargs = mock_ann.return_value.annotate.call_args
    passed_preds = mock_ann.return_value.annotate.call_args[0][2]
    assert passed_preds == [pred]


# ---------------------------------------------------------------------------
# run() — empty frame list
# ---------------------------------------------------------------------------


def test_empty_frame_list_returns_empty_result() -> None:
    """Zero sampled frames must return a PipelineResult with empty lists."""
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR),
        patch(_PATCH_TRACKER),
        patch(_PATCH_ANNOTATOR),
        patch(_PATCH_TIMELINE),
        patch(_PATCH_GRAPH),
    ):
        metadata = _make_metadata(0)
        mock_ing.return_value.load.return_value = ([], metadata)
        pipeline = CrowdFlowPipeline()
        result = pipeline.run("fake.mp4")

    assert result.annotated_frames == []
    assert result.timeline == []
    assert result.metadata == metadata


# ---------------------------------------------------------------------------
# run() — error propagation
# ---------------------------------------------------------------------------


def test_ingestion_error_propagates() -> None:
    """CrowdFlowError subclasses from ingestion must propagate unchanged."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR),
        patch(_PATCH_TRACKER),
        patch(_PATCH_ANNOTATOR),
        patch(_PATCH_TIMELINE),
        patch(_PATCH_GRAPH),
    ):
        mock_ing.return_value.load.side_effect = VideoCorruptionError("bad file")
        pipeline = CrowdFlowPipeline()
        with pytest.raises(VideoCorruptionError):
            pipeline.run("bad.mp4")


def test_detection_error_propagates() -> None:
    """ModelInferenceError from detection must propagate unchanged."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER),
        patch(_PATCH_ANNOTATOR),
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH),
    ):
        metadata = _make_metadata(1)
        mock_ing.return_value.load.return_value = ([_make_frame()], metadata)
        mock_det.return_value.detect.side_effect = ModelInferenceError("yolo fail")
        mock_tl.return_value.reset.return_value = None
        pipeline = CrowdFlowPipeline()
        with pytest.raises(ModelInferenceError):
            pipeline.run("fake.mp4")


def test_unexpected_frame_error_wrapped_as_crowdflow_error() -> None:
    """Unexpected non-CrowdFlow exceptions inside the frame loop must be
    caught and re-raised as CrowdFlowError."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER),
        patch(_PATCH_ANNOTATOR),
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH),
    ):
        metadata = _make_metadata(1)
        mock_ing.return_value.load.return_value = ([_make_frame()], metadata)
        mock_det.return_value.detect.side_effect = RuntimeError("unexpected")
        mock_tl.return_value.reset.return_value = None
        pipeline = CrowdFlowPipeline()
        with pytest.raises(CrowdFlowError, match="Unexpected error on frame"):
            pipeline.run("fake.mp4")


def test_crowdflow_error_propagates_unchanged() -> None:
    """A CrowdFlowError raised inside the loop must propagate without wrapping."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER),
        patch(_PATCH_ANNOTATOR),
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH),
    ):
        metadata = _make_metadata(1)
        original = CrowdFlowError("draw fail")
        mock_ing.return_value.load.return_value = ([_make_frame()], metadata)
        mock_det.return_value.detect.side_effect = original
        mock_tl.return_value.reset.return_value = None
        pipeline = CrowdFlowPipeline()
        with pytest.raises(CrowdFlowError) as exc_info:
            pipeline.run("fake.mp4")
        assert exc_info.value is original


# ---------------------------------------------------------------------------
# run() — graph/model ImportError (torch_geometric not installed)
# ---------------------------------------------------------------------------


def test_torch_geometric_import_error_handled_gracefully() -> None:
    """If torch_geometric is not installed, the frame must still be annotated
    with empty predictions — no exception raised."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    mock_model = MagicMock(return_value=[_make_pred()])
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH)    as mock_gb,
    ):
        frames = [_make_frame()]
        metadata = _make_metadata(1)
        mock_ing.return_value.load.return_value = (frames, metadata)
        mock_det.return_value.detect.return_value = [(0.0, 0.0, 50.0, 50.0, 0.9, 0)]
        mock_trk.return_value.update.return_value = [_make_track()]
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = []
        mock_gb.return_value.build.side_effect = ImportError("torch_geometric not installed")

        pipeline = CrowdFlowPipeline(model_fn=mock_model)
        result = pipeline.run("fake.mp4")

    assert len(result.annotated_frames) == 1
    # model_fn should not have been called since build raised ImportError
    mock_model.assert_not_called()


# ---------------------------------------------------------------------------
# _extract_arrays() — unit tests
# ---------------------------------------------------------------------------


def test_extract_arrays_shape() -> None:
    """_extract_arrays must return (N,2) arrays for N tracks."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    tracks = [_make_track(1, 100.0, 200.0), _make_track(2, 300.0, 400.0)]
    pos, vel = CrowdFlowPipeline._extract_arrays(tracks, _W, _H)
    assert pos.shape == (2, 2)
    assert vel.shape == (2, 2)


def test_extract_arrays_normalises_positions() -> None:
    """Positions must be centroid/frame_dim, i.e. in [0, 1]."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    track = _make_track(1, cx=320.0, cy=240.0)  # exact centre of 640x480
    pos, _ = CrowdFlowPipeline._extract_arrays([track], _W, _H)
    assert pos[0, 0] == pytest.approx(0.5)   # cx / width
    assert pos[0, 1] == pytest.approx(0.5)   # cy / height


def test_extract_arrays_velocity_not_normalised() -> None:
    """Velocities are passed as raw pixel deltas, not normalised."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    track = TrackItem(
        track_id=1,
        bbox=(0.0, 0.0, 50.0, 50.0),
        centroid=(25.0, 25.0),
        velocity=(10.0, -5.0),
    )
    _, vel = CrowdFlowPipeline._extract_arrays([track], _W, _H)
    assert vel[0, 0] == pytest.approx(10.0)
    assert vel[0, 1] == pytest.approx(-5.0)


def test_extract_arrays_dtype_is_float32() -> None:
    """Both returned arrays must have dtype float32 (GraphBuilder contract)."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    tracks = [_make_track()]
    pos, vel = CrowdFlowPipeline._extract_arrays(tracks, _W, _H)
    assert pos.dtype == np.float32
    assert vel.dtype == np.float32


def test_extract_arrays_multiple_tracks_ordering() -> None:
    """Track ordering in the output arrays must match input list ordering."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    t1 = _make_track(1, cx=100.0, cy=200.0)
    t2 = _make_track(2, cx=400.0, cy=300.0)
    pos, _ = CrowdFlowPipeline._extract_arrays([t1, t2], _W, _H)
    assert pos[0, 0] == pytest.approx(100.0 / _W)
    assert pos[1, 0] == pytest.approx(400.0 / _W)


# ---------------------------------------------------------------------------
# GraphBuilder proximity radius normalisation
# ---------------------------------------------------------------------------


def test_proximity_radius_clamped_when_too_large() -> None:
    """When PROXIMITY_RADIUS > min(w,h), norm_radius must be clamped to 1.0
    and GraphBuilder must still be constructed without ValueError."""
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    with (
        patch(_PATCH_INGESTOR) as mock_ing,
        patch(_PATCH_DETECTOR) as mock_det,
        patch(_PATCH_TRACKER)  as mock_trk,
        patch(_PATCH_ANNOTATOR) as mock_ann,
        patch(_PATCH_TIMELINE) as mock_tl,
        patch(_PATCH_GRAPH)    as mock_gb,
        patch("crowdflow_dna.pipeline.config") as mock_cfg,
    ):
        # Frame smaller than PROXIMITY_RADIUS
        tiny_meta = {
            "fps": 30.0, "width": 10, "height": 10,
            "frame_count": 5, "duration_seconds": 1.0, "sample_rate": 5,
        }
        mock_ing.return_value.load.return_value = ([_make_frame()], tiny_meta)
        mock_det.return_value.detect.return_value = []
        mock_trk.return_value.update.return_value = []
        mock_ann.return_value.annotate.side_effect = lambda f, t, p: f.copy()
        mock_tl.return_value.get_timeline.return_value = []
        mock_cfg.PROXIMITY_RADIUS = 50.0  # 50 >> min(10,10)
        mock_cfg.FRAME_SAMPLE_RATE = 5

        pipeline = CrowdFlowPipeline()
        result = pipeline.run("fake.mp4")

    # GraphBuilder must be constructed with proximity_radius=1.0
    mock_gb.assert_called_once_with(proximity_radius=1.0)
    assert len(result.annotated_frames) == 1
