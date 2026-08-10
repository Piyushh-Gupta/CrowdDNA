"""Phase 9 final integration smoke tests — deployment.pt.

These tests load the *real* ``deployment.pt`` artifact to verify that the
full inference pipeline works end-to-end without mocks.  They are skipped
automatically in CI (where the artifact is not present) by the ``skip_no_model``
fixture.

Run locally with::

    $env:CROWDDNA_MODEL_PATH = "C:\\path\\to\\deployment.pt"
    pytest tests/test_phase9_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure the project root is on sys.path regardless of pytest invocation method.
# This mirrors the pattern used in test_app.py and other test modules.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

try:
    from torch_geometric.data import Data
except ImportError:
    pytest.skip(
        "torch_geometric not installed — skipping Phase 9 integration tests.",
        allow_module_level=True,
    )

from crowdflow_dna.inference import (
    InferenceResult,
    InferenceRuntime,
    SequenceBuffer,
    TensorBatch,
)
from crowdflow_dna.pipeline import CrowdFlowPipeline
from crowdflow_dna.schemas import TrackItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_path() -> Path:
    """Return the path to deployment.pt, skipping if not present."""
    env = os.environ.get("CROWDDNA_MODEL_PATH")
    if env and Path(env).is_file():
        return Path(env)
    # Fallback: look for deployment.pt at the repository root
    repo_root = Path(__file__).parent.parent
    candidate = repo_root / "deployment.pt"
    if candidate.is_file():
        return candidate
    pytest.skip(
        "deployment.pt not found — set CROWDDNA_MODEL_PATH or place the file "
        "at the repository root. Skipping Phase 9 integration tests."
    )


@pytest.fixture(scope="module")
def runtime(model_path: Path) -> InferenceRuntime:
    """Load InferenceRuntime with the real deployment.pt once per session."""
    rt = InferenceRuntime()
    rt.load_model(model_path)
    return rt


# ---------------------------------------------------------------------------
# Helper: build a synthetic TensorBatch
# ---------------------------------------------------------------------------


def _make_tensor_batch(
    n_nodes: int = 6,
    n_edges: int = 8,
    window_size: int = 3,
) -> TensorBatch:
    """Assemble a synthetic single-sequence TensorBatch via SequenceBuffer."""
    buf = SequenceBuffer(window_size=window_size)
    per_frame = max(1, n_nodes // window_size)
    for _ in range(window_size):
        x = torch.rand(per_frame, 5, dtype=torch.float32)
        if per_frame > 1:
            ei = torch.randint(0, per_frame, (2, 2), dtype=torch.long)
            ea = torch.rand(2, 4, dtype=torch.float32)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 4), dtype=torch.float32)
        buf.push(Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=per_frame))
    assert buf.is_ready
    return buf.assemble()


# ---------------------------------------------------------------------------
# InferenceRuntime smoke tests
# ---------------------------------------------------------------------------


class TestInferenceRuntimeSmoke:
    """Verify InferenceRuntime loads and predicts correctly with the real model."""

    def test_model_loaded(self, runtime: InferenceRuntime) -> None:
        assert runtime.backend is not None
        assert runtime.backend_name == "TorchScript"
        assert runtime.model_format == "TorchScript"

    def test_predict_returns_inference_result(self, runtime: InferenceRuntime) -> None:
        tb = _make_tensor_batch()
        result = runtime.predict(
            tb.x, tb.edge_index, tb.edge_attr, tb.batch, tb.seq_lengths
        )
        assert isinstance(result, InferenceResult)

    def test_predicted_class_is_valid(self, runtime: InferenceRuntime) -> None:
        tb = _make_tensor_batch()
        result = runtime.predict(
            tb.x, tb.edge_index, tb.edge_attr, tb.batch, tb.seq_lengths
        )
        assert result.predicted_class in (0, 1, 2), (
            f"Expected class index 0-2, got {result.predicted_class}"
        )

    def test_confidence_is_probability(self, runtime: InferenceRuntime) -> None:
        tb = _make_tensor_batch()
        result = runtime.predict(
            tb.x, tb.edge_index, tb.edge_attr, tb.batch, tb.seq_lengths
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self, runtime: InferenceRuntime) -> None:
        tb = _make_tensor_batch()
        result = runtime.predict(
            tb.x, tb.edge_index, tb.edge_attr, tb.batch, tb.seq_lengths
        )
        total = float(np.sum(result.probabilities))
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    def test_inference_latency_within_budget(self, runtime: InferenceRuntime) -> None:
        """Per-call latency must stay < 35 ms (contract §9)."""
        import time

        tb = _make_tensor_batch()
        start = time.perf_counter()
        runtime.predict(
            tb.x, tb.edge_index, tb.edge_attr, tb.batch, tb.seq_lengths
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms < 35.0, (
            f"Inference latency {elapsed_ms:.2f} ms exceeds 35 ms budget"
        )


# ---------------------------------------------------------------------------
# Pipeline metadata injection test
# ---------------------------------------------------------------------------


class TestPipelineMetadataInjection:
    """Verify that pipeline.run() injects runtime provenance into metadata."""

    def _build_pipeline(self, model_path: Path) -> CrowdFlowPipeline:
        """Build a pipeline where only the ingestor/detector/tracker are mocked."""
        pipeline = CrowdFlowPipeline(model_path=str(model_path), window_size=2)

        # Replace heavy I/O components with lightweight mocks
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
        metadata = {
            "fps": 30.0,
            "width": 64,
            "height": 64,
            "frame_count": 4,
            "duration_seconds": 4 / 30.0,
            "sample_rate": 1,
        }
        pipeline._ingestor = MagicMock()
        pipeline._ingestor.load.return_value = (frames, metadata)

        # Detector returns nothing; tracker returns one track per frame
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []
        pipeline._tracker = MagicMock()
        pipeline._tracker.update.return_value = [
            TrackItem(
                track_id=1,
                bbox=(0.0, 0.0, 32.0, 32.0),
                centroid=(16.0, 16.0),
                velocity=(0.0, 0.0),
            )
        ]

        # Keep the real annotator and timeline for full coverage
        return pipeline

    def test_metadata_contains_backend(self, model_path: Path) -> None:
        pipeline = self._build_pipeline(model_path)
        with patch("subprocess.Popen"), patch("shutil.which", return_value="ffmpeg"):
            result = pipeline.run("dummy.mp4")
        assert "backend" in result.metadata
        assert result.metadata["backend"] == "TorchScript"

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="ffmpeg")
    def test_end_to_end_runtime_integration(
        self, mock_which, mock_popen, model_path: Path
    ) -> None:
        pipeline = self._build_pipeline(model_path)
        result = pipeline.run("dummy.mp4")
        assert result.metadata.get("model_format") == "TorchScript"

    def test_metadata_contains_model_version(self, model_path: Path) -> None:
        pipeline = self._build_pipeline(model_path)
        result = pipeline.run("dummy.mp4")
        # version may be None when not passed, but key must exist
        assert "model_version" in result.metadata

    def test_dummy_mode_has_no_runtime_keys(self) -> None:
        """Dummy-mode results must NOT include backend/model_format keys."""
        pipeline = CrowdFlowPipeline(model_path=None)
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        meta = {
            "fps": 30.0, "width": 64, "height": 64,
            "frame_count": 2, "duration_seconds": 2 / 30.0, "sample_rate": 1,
        }
        pipeline._ingestor = MagicMock()
        pipeline._ingestor.load.return_value = (frames, meta)
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []
        pipeline._tracker = MagicMock()
        pipeline._tracker.update.return_value = []

        with patch("subprocess.Popen"), patch("shutil.which", return_value="ffmpeg"):
            result = pipeline.run("dummy.mp4")
        assert "backend" not in result.metadata
        assert "model_format" not in result.metadata
