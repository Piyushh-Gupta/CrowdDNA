"""Integration tests for Phase 12B: pipeline + inference runtime integration.

Tests cover:
- SequenceBuffer assembly (warm-up, full window, empty frames, offset correctness)
- InferenceResult → RiskPrediction translation
- CrowdFlowPipeline in dummy mode (unchanged behaviour)
- CrowdFlowPipeline in inference mode (end-to-end with mocked runtime)
- Pipeline behaviour when InferenceRuntime raises InferenceExecutionError
- Pipeline behaviour with model path that doesn't exist
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from crowdflow_dna.inference import (
    InferenceExecutionError,
    InferenceResult,
    ModelNotFoundError,
    SequenceBuffer,
    TensorBatch,
    UnsupportedModelFormatError,
)
from crowdflow_dna.errors import ModelInferenceError
from crowdflow_dna.pipeline import CrowdFlowPipeline, PipelineResult
from crowdflow_dna.schemas import TrackItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_nodes: int = 3, n_edges: int = 4) -> Data:
    """Build a minimal synthetic PyG Data graph."""
    x = torch.rand(n_nodes, 5, dtype=torch.float32)
    if n_nodes > 1 and n_edges > 0:
        edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)
        edge_attr = torch.rand(n_edges, 4, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_nodes,
    )


def _empty_data() -> Data:
    """Build a zero-node placeholder graph (contract §6.4)."""
    return Data(
        x=torch.zeros((0, 5), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 4), dtype=torch.float32),
        num_nodes=0,
    )


def _make_track(track_id: int = 1) -> TrackItem:
    return TrackItem(
        track_id=track_id,
        bbox=(10.0, 20.0, 50.0, 80.0),
        centroid=(30.0, 50.0),
        velocity=(1.0, 0.5),
    )


def _mock_inference_result(predicted_class: int = 0) -> InferenceResult:
    probs = np.zeros(max(3, predicted_class + 1), dtype=np.float64)
    probs[predicted_class] = 1.0
    return InferenceResult(
        predicted_class=predicted_class,
        probabilities=probs,
        confidence=1.0,
        backend="TorchScript",
        inference_time_ms=1.0,
        model_format="TorchScript",
        model_version="test",
    )


# ---------------------------------------------------------------------------
# SequenceBuffer tests
# ---------------------------------------------------------------------------


class TestSequenceBuffer:
    def test_not_ready_until_window_full(self):
        buf = SequenceBuffer(window_size=3)
        assert not buf.is_ready
        buf.push(_make_data())
        assert not buf.is_ready
        buf.push(_make_data())
        assert not buf.is_ready
        buf.push(_make_data())
        assert buf.is_ready

    def test_assemble_raises_if_not_ready(self):
        buf = SequenceBuffer(window_size=5)
        buf.push(_make_data())
        with pytest.raises(RuntimeError, match="assemble"):
            buf.assemble()

    def test_assemble_returns_correct_batch(self):
        buf = SequenceBuffer(window_size=2)
        buf.push(_make_data(n_nodes=3, n_edges=4))
        buf.push(_make_data(n_nodes=2, n_edges=2))
        tb = buf.assemble()

        assert isinstance(tb, TensorBatch)
        assert tb.x.shape == (5, 5)              # 3+2 nodes, 5 features
        assert tb.edge_index.shape[0] == 2
        assert tb.edge_attr.shape[1] == 4
        assert tb.batch.shape[0] == 5             # 3+2 nodes
        assert tb.seq_lengths.tolist() == [2]

    def test_edge_index_offset_applied_correctly(self):
        """Nodes in frame 1 should be offset by the number of nodes in frame 0."""
        buf = SequenceBuffer(window_size=2)
        data0 = _make_data(n_nodes=3, n_edges=2)
        data1 = _make_data(n_nodes=2, n_edges=2)
        buf.push(data0)
        buf.push(data1)
        tb = buf.assemble()

        total_nodes = 5
        assert tb.edge_index.max().item() < total_nodes
        # Edges from frame 1 should reference global indices >= 3 (offset by n0=3)
        n_edges_0 = data0.edge_index.shape[1]
        if n_edges_0 < tb.edge_index.shape[1]:
            frame1_edges = tb.edge_index[:, n_edges_0:]
            assert frame1_edges.min().item() >= 3

    def test_empty_frame_handled_gracefully(self):
        buf = SequenceBuffer(window_size=3)
        buf.push(_make_data(n_nodes=3, n_edges=2))
        buf.push(_empty_data())               # zero-node frame -> gets 1 dummy node
        buf.push(_make_data(n_nodes=2, n_edges=2))
        assert buf.is_ready

        tb = buf.assemble()
        assert tb.x.shape == (6, 5)           # 3 + 1(dummy) + 2 = 6 nodes
        assert tb.seq_lengths.tolist() == [3]  # still 3 frames

    def test_all_empty_frames(self):
        buf = SequenceBuffer(window_size=2)
        buf.push(_empty_data())
        buf.push(_empty_data())
        tb = buf.assemble()

        assert tb.x.shape == (2, 5)           # 2 dummy nodes
        assert tb.edge_index.shape == (2, 0)
        assert tb.edge_attr.shape == (0, 4)
        assert tb.batch.shape == (2,)         # 2 items in batch
        assert tb.seq_lengths.tolist() == [2]

    def test_reset_clears_buffer(self):
        buf = SequenceBuffer(window_size=2)
        buf.push(_make_data())
        buf.push(_make_data())
        assert buf.is_ready
        buf.reset()
        assert not buf.is_ready

    def test_window_slides_on_overflow(self):
        """After filling, the oldest frame is evicted."""
        buf = SequenceBuffer(window_size=2)
        data0 = _make_data(n_nodes=3, n_edges=0)
        data1 = _make_data(n_nodes=4, n_edges=0)
        data2 = _make_data(n_nodes=5, n_edges=0)

        buf.push(data0)
        buf.push(data1)
        buf.push(data2)  # data0 evicted; window = [data1, data2]

        tb = buf.assemble()
        # 4+5=9 nodes (data0's 3 nodes should NOT appear)
        assert tb.x.shape == (9, 5)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            SequenceBuffer(window_size=0)


# ---------------------------------------------------------------------------
# InferenceResult → RiskPrediction mapping tests
# ---------------------------------------------------------------------------


class TestResultToPredictions:
    def test_maps_to_all_tracks(self):
        result = _mock_inference_result(predicted_class=2)  # Critical
        tracks = [_make_track(1), _make_track(2)]

        preds = CrowdFlowPipeline._result_to_predictions(result, tracks)

        assert len(preds) == 2
        assert all(p.label == "Critical" for p in preds)
        assert all(p.confidence == pytest.approx(1.0) for p in preds)
        assert {p.region_id for p in preds} == {1, 2}

    def test_returns_empty_when_no_tracks(self):
        result = _mock_inference_result(predicted_class=0)
        preds = CrowdFlowPipeline._result_to_predictions(result, [])
        assert preds == []

    def test_all_risk_labels(self):
        tracks = [_make_track()]
        for cls_idx, expected_label in enumerate(["Safe", "Congesting", "Critical"]):
            result = _mock_inference_result(predicted_class=cls_idx)
            preds = CrowdFlowPipeline._result_to_predictions(result, tracks)
            assert preds[0].label == expected_label

    def test_raises_model_inference_error_on_invalid_predicted_class(self):
        result = _mock_inference_result(predicted_class=99)
        tracks = [_make_track()]
        with pytest.raises(ModelInferenceError, match="invalid predicted_class index"):
            CrowdFlowPipeline._result_to_predictions(result, tracks)


# ---------------------------------------------------------------------------
# Pipeline dummy-mode tests (no model)
# ---------------------------------------------------------------------------


class TestPipelineDummyMode:
    def _make_pipeline(self) -> CrowdFlowPipeline:
        """Return a dummy-mode pipeline with all components mocked."""
        with patch("crowdflow_dna.pipeline.VideoIngestor"), \
             patch("crowdflow_dna.pipeline.Yolov8Detector"), \
             patch("crowdflow_dna.pipeline.ByteTracker"), \
             patch("crowdflow_dna.pipeline.FrameAnnotator"), \
             patch("crowdflow_dna.pipeline.TimelineBuilder"):
            return CrowdFlowPipeline(model_path=None)

    def test_instantiation_dummy_mode(self):
        p = self._make_pipeline()
        assert p._runtime is None
        assert p._seq_buffer is None

    def test_runtime_loaded_on_valid_model_path(self, tmp_path):
        """Providing a real .pt file that exists should trigger runtime load."""
        fake_model = tmp_path / "model.pt"
        fake_model.touch()

        mock_runtime = MagicMock()
        with patch("crowdflow_dna.pipeline.InferenceRuntime", return_value=mock_runtime):
            p = CrowdFlowPipeline(model_path=str(fake_model))

        mock_runtime.load_model.assert_called_once_with(
            str(fake_model), version=None
        )
        assert p._runtime is mock_runtime

    def test_raises_on_missing_model_path(self):
        with pytest.raises(ModelNotFoundError):
            CrowdFlowPipeline(model_path="/non/existent/model.pt")

    def test_raises_on_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "model.h5"
        bad_file.touch()
        with pytest.raises(UnsupportedModelFormatError):
            CrowdFlowPipeline(model_path=str(bad_file))


# ---------------------------------------------------------------------------
# End-to-end pipeline with mocked dependencies
# ---------------------------------------------------------------------------


class TestPipelineInferenceIntegration:
    """End-to-end tests using fully mocked I/O, detector, tracker, and runtime."""

    def _build_pipeline_with_mocked_runtime(
        self,
        mock_predict_return: InferenceResult,
        window_size: int = 3,
    ) -> CrowdFlowPipeline:
        """Construct a pipeline where InferenceRuntime.predict returns a fixed result."""
        mock_runtime = MagicMock()
        mock_runtime.predict.return_value = mock_predict_return

        pipeline = CrowdFlowPipeline.__new__(CrowdFlowPipeline)
        pipeline._ingestor = MagicMock()
        pipeline._detector = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._annotator = MagicMock()
        pipeline._timeline = MagicMock()
        pipeline._runtime = mock_runtime
        pipeline._seq_buffer = SequenceBuffer(window_size=window_size)
        return pipeline

    def _configure_ingestor(self, pipeline, n_frames: int = 12):
        """Point the mock ingestor at n_frames of black 100×100 frames."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(n_frames)]
        metadata = {
            "fps": 30.0, "width": 100, "height": 100,
            "frame_count": n_frames, "duration_seconds": n_frames / 30.0,
            "sample_rate": 1,
        }
        pipeline._ingestor.load.return_value = (frames, metadata)
        return frames, metadata

    def test_inference_fires_after_warmup(self):
        """predict() must be called only after window_size frames are buffered."""
        inf_result = _mock_inference_result(0)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=3)
        frames, metadata = self._configure_ingestor(pipeline, n_frames=5)

        # Make tracker return 2 tracks per frame so GraphBuilder has nodes
        tracks = [_make_track(1), _make_track(2)]
        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = tracks
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        pipeline.run("dummy.mp4")

        # deque(maxlen=3): ready from frame index 2 onwards → 3 predict calls (frames 2, 3, 4)
        assert pipeline._runtime.predict.call_count == 3

    def test_warmup_frames_produce_no_predictions(self):
        """Frames before the buffer is full must return empty predictions."""
        inf_result = _mock_inference_result(1)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=10)
        frames, _ = self._configure_ingestor(pipeline, n_frames=5)

        tracks = [_make_track(1)]
        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = tracks
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        pipeline.run("dummy.mp4")
        pipeline._runtime.predict.assert_not_called()

    def test_empty_track_frame_does_not_crash(self):
        """A frame with zero tracks must not raise."""
        inf_result = _mock_inference_result(0)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=2)
        frames, _ = self._configure_ingestor(pipeline, n_frames=3)

        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = []  # zero tracks every frame
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        # Must not raise
        pipeline.run("dummy.mp4")

    def test_inference_error_raises_model_inference_error(self):
        """InferenceExecutionError from the runtime must propagate as ModelInferenceError."""
        from crowdflow_dna.errors import ModelInferenceError

        inf_result = _mock_inference_result(0)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=1)
        frames, _ = self._configure_ingestor(pipeline, n_frames=2)

        pipeline._runtime.predict.side_effect = InferenceExecutionError("bad tensor")
        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = [_make_track()]
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        with pytest.raises(ModelInferenceError, match="InferenceRuntime failed"):
            pipeline.run("dummy.mp4")

    def test_risk_predictions_reach_annotator(self):
        """Verify that RiskPrediction objects are forwarded to FrameAnnotator."""
        inf_result = _mock_inference_result(2)  # Critical
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=1)
        frames, _ = self._configure_ingestor(pipeline, n_frames=2)

        tracks = [_make_track(1)]
        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = tracks
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        pipeline.run("dummy.mp4")

        # After warm-up (frame 0), frame 1 produces predictions
        call_args = pipeline._annotator.annotate.call_args_list
        # At least one call must have passed a non-empty predictions list
        any_with_preds = any(
            args[0][2] and len(args[0][2]) > 0
            for args in call_args
        )
        assert any_with_preds

        # All non-empty prediction calls must label "Critical"
        for args in call_args:
            preds: list = args[0][2]
            for p in preds:
                assert p.label == "Critical"

    def test_pipeline_result_structure(self):
        """PipelineResult must contain frames, timeline, and metadata."""
        inf_result = _mock_inference_result(0)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=2)
        n_frames = 4
        frames, metadata = self._configure_ingestor(pipeline, n_frames=n_frames)

        pipeline._detector.detect.return_value = []
        pipeline._tracker.update.return_value = [_make_track()]
        pipeline._annotator.annotate.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        pipeline._timeline.reset = MagicMock()
        pipeline._timeline.record = MagicMock()
        pipeline._timeline.get_timeline.return_value = []

        result = pipeline.run("dummy.mp4")

        assert isinstance(result, PipelineResult)
        assert len(result.annotated_frames) == n_frames
        assert result.metadata["fps"] == pytest.approx(30.0)

    def test_buffer_reset_between_runs(self):
        """Calling run() twice should reset the buffer so warm-up restarts."""
        inf_result = _mock_inference_result(0)
        pipeline = self._build_pipeline_with_mocked_runtime(inf_result, window_size=3)

        for _ in range(2):
            frames, _ = self._configure_ingestor(pipeline, n_frames=2)
            pipeline._detector.detect.return_value = []
            pipeline._tracker.update.return_value = [_make_track()]
            pipeline._annotator.annotate.return_value = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )
            pipeline._timeline.reset = MagicMock()
            pipeline._timeline.record = MagicMock()
            pipeline._timeline.get_timeline.return_value = []
            pipeline.run("dummy.mp4")

        # With window_size=3 and only 2 frames each run, predict must never fire
        pipeline._runtime.predict.assert_not_called()
