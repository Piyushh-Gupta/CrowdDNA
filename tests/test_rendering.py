"""Unit tests for the Rendering module.

Covers FrameAnnotator (video_renderer.py) and TimelineBuilder (timeline.py).
All tests are deterministic and require no GPU or real video files.
OpenCV drawing calls are exercised on small synthetic frames.
"""

import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.rendering.timeline import TimelineBuilder, TimelineEntry
from crowdflow_dna.rendering.video_renderer import FrameAnnotator, _DEFAULT_COLOUR
from crowdflow_dna.schemas import RiskPrediction, TrackItem

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A small blank BGR frame used in all FrameAnnotator tests.
_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)


def _track(
    track_id: int = 1,
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 300.0,
) -> TrackItem:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return TrackItem(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        centroid=(cx, cy),
        velocity=(0.0, 0.0),
    )


def _pred(
    region_id: int = 1,
    label: str = "Safe",
    confidence: float = 0.9,
) -> RiskPrediction:
    return RiskPrediction(region_id=region_id, label=label, confidence=confidence)


# ---------------------------------------------------------------------------
# FrameAnnotator — return type and shape
# ---------------------------------------------------------------------------


@pytest.fixture()
def annotator() -> FrameAnnotator:
    return FrameAnnotator()


def test_annotate_returns_ndarray(annotator: FrameAnnotator) -> None:
    """annotate() must always return a numpy ndarray."""
    result = annotator.annotate(_FRAME, [_track()], [_pred()])
    assert isinstance(result, np.ndarray)


def test_annotate_output_shape_matches_input(annotator: FrameAnnotator) -> None:
    """Output frame must have the same shape as the input frame."""
    result = annotator.annotate(_FRAME, [_track()], [_pred()])
    assert result.shape == _FRAME.shape


def test_annotate_output_dtype_matches_input(annotator: FrameAnnotator) -> None:
    """Output frame must have the same dtype as the input frame."""
    result = annotator.annotate(_FRAME, [_track()], [_pred()])
    assert result.dtype == _FRAME.dtype


# ---------------------------------------------------------------------------
# FrameAnnotator — input immutability
# ---------------------------------------------------------------------------


def test_annotate_does_not_mutate_input_frame(annotator: FrameAnnotator) -> None:
    """annotate() must return a copy; the original frame must be unchanged."""
    original = np.zeros((480, 640, 3), dtype=np.uint8)
    snapshot = original.copy()
    annotator.annotate(original, [_track()], [_pred()])
    np.testing.assert_array_equal(original, snapshot)


# ---------------------------------------------------------------------------
# FrameAnnotator — empty input cases
# ---------------------------------------------------------------------------


def test_annotate_empty_tracks_returns_frame(annotator: FrameAnnotator) -> None:
    """Empty track list must return a frame (equal to input copy), no error."""
    result = annotator.annotate(_FRAME, [], [])
    assert isinstance(result, np.ndarray)
    assert result.shape == _FRAME.shape


def test_annotate_empty_tracks_no_exception(annotator: FrameAnnotator) -> None:
    """Empty tracks must not raise any exception."""
    try:
        annotator.annotate(_FRAME, [], [_pred()])
    except Exception as exc:
        pytest.fail(f"annotate() raised unexpectedly with empty tracks: {exc}")


def test_annotate_empty_predictions_no_exception(annotator: FrameAnnotator) -> None:
    """Tracks with no matching predictions must not raise any exception."""
    try:
        annotator.annotate(_FRAME, [_track()], [])
    except Exception as exc:
        pytest.fail(f"annotate() raised unexpectedly with empty predictions: {exc}")


# ---------------------------------------------------------------------------
# FrameAnnotator — colour application
# ---------------------------------------------------------------------------


def test_annotate_safe_label_applies_green_colour(annotator: FrameAnnotator) -> None:
    """A 'Safe' prediction must result in a green (0, 200, 0) bbox pixel."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = annotator.annotate(frame, [_track(1, 100, 100, 200, 300)], [_pred(1, "Safe")])
    # Sample a pixel on the left edge of the drawn box (x=100, y=200 midpoint)
    b, g, r = result[200, 100]
    assert g > 0, "Safe label should produce green pixels on the box edge"
    assert r == 0
    assert b == 0


def test_annotate_congesting_label_applies_orange_colour(annotator: FrameAnnotator) -> None:
    """A 'Congesting' prediction must result in an orange bbox pixel."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = annotator.annotate(
        frame, [_track(1, 100, 100, 200, 300)], [_pred(1, "Congesting")]
    )
    b, g, r = result[200, 100]
    # BGR orange = (0, 165, 255) — blue=0, green>0, red=255
    assert r == 255
    assert g > 0
    assert b == 0


def test_annotate_critical_label_applies_red_colour(annotator: FrameAnnotator) -> None:
    """A 'Critical' prediction must result in a red bbox pixel."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = annotator.annotate(
        frame, [_track(1, 100, 100, 200, 300)], [_pred(1, "Critical")]
    )
    b, g, r = result[200, 100]
    # BGR red = (0, 0, 220)
    assert r == 220
    assert g == 0
    assert b == 0


def test_annotate_unmatched_track_uses_default_colour(annotator: FrameAnnotator) -> None:
    """A track with no matching prediction must use the default grey colour."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # track_id=1, prediction region_id=99 — no match
    result = annotator.annotate(
        frame, [_track(1, 100, 100, 200, 300)], [_pred(99, "Safe")]
    )
    b, g, r = result[200, 100]
    expected_b, expected_g, expected_r = _DEFAULT_COLOUR
    assert b == expected_b
    assert g == expected_g
    assert r == expected_r


def test_annotate_unknown_risk_label_uses_default_colour(
    annotator: FrameAnnotator,
) -> None:
    """An unrecognised risk label string falls back to the default colour."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = annotator.annotate(
        frame, [_track(1, 100, 100, 200, 300)], [_pred(1, "Unknown")]
    )
    b, g, r = result[200, 100]
    expected_b, expected_g, expected_r = _DEFAULT_COLOUR
    assert b == expected_b
    assert g == expected_g
    assert r == expected_r


# ---------------------------------------------------------------------------
# FrameAnnotator — multiple tracks
# ---------------------------------------------------------------------------


def test_annotate_multiple_tracks_all_drawn(annotator: FrameAnnotator) -> None:
    """All tracks must be drawn; output must differ from the blank input."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = [
        _track(1, 10, 10, 80, 80),
        _track(2, 200, 200, 300, 350),
        _track(3, 400, 50, 500, 150),
    ]
    preds = [_pred(1, "Safe"), _pred(2, "Congesting"), _pred(3, "Critical")]
    result = annotator.annotate(frame, tracks, preds)
    # At least some pixels must be non-zero (boxes drawn)
    assert result.any(), "Annotated frame should have non-zero pixels"


# ---------------------------------------------------------------------------
# FrameAnnotator — error handling
# ---------------------------------------------------------------------------


def test_annotate_opencv_error_raises_crowdflow_error(
    annotator: FrameAnnotator,
) -> None:
    """If OpenCV drawing raises, CrowdFlowError must propagate."""
    with patch("crowdflow_dna.rendering.video_renderer.cv2.rectangle") as mock_rect:
        mock_rect.side_effect = RuntimeError("cv2 error")
        with pytest.raises(CrowdFlowError, match="FrameAnnotator failed"):
            annotator.annotate(_FRAME, [_track()], [])


def test_annotate_error_chains_original_exception(annotator: FrameAnnotator) -> None:
    """CrowdFlowError must chain the original exception via __cause__."""
    original = ValueError("bad frame shape")
    with patch("crowdflow_dna.rendering.video_renderer.cv2.rectangle") as mock_rect:
        mock_rect.side_effect = original
        with pytest.raises(CrowdFlowError) as exc_info:
            annotator.annotate(_FRAME, [_track()], [])
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# TimelineEntry — dataclass contract
# ---------------------------------------------------------------------------


def test_timeline_entry_has_correct_fields() -> None:
    """TimelineEntry must have exactly frame_index and predictions fields."""
    field_names = {f.name for f in fields(TimelineEntry)}
    assert field_names == {"frame_index", "predictions"}


def test_timeline_entry_stores_frame_index() -> None:
    """TimelineEntry.frame_index must match the value provided."""
    entry = TimelineEntry(frame_index=42, predictions=[])
    assert entry.frame_index == 42


def test_timeline_entry_stores_predictions() -> None:
    """TimelineEntry.predictions must match the list provided."""
    preds = [_pred(1, "Safe"), _pred(2, "Critical")]
    entry = TimelineEntry(frame_index=0, predictions=preds)
    assert entry.predictions == preds


def test_timeline_entry_defaults_to_empty_predictions() -> None:
    """TimelineEntry should accept an empty predictions list as default."""
    entry = TimelineEntry(frame_index=0)
    assert entry.predictions == []


# ---------------------------------------------------------------------------
# TimelineBuilder — record and get_timeline
# ---------------------------------------------------------------------------


@pytest.fixture()
def builder() -> TimelineBuilder:
    return TimelineBuilder()


def test_get_timeline_returns_list(builder: TimelineBuilder) -> None:
    """get_timeline() must always return a list."""
    assert isinstance(builder.get_timeline(), list)


def test_get_timeline_empty_on_init(builder: TimelineBuilder) -> None:
    """A fresh TimelineBuilder must have an empty timeline."""
    assert builder.get_timeline() == []


def test_record_adds_entry(builder: TimelineBuilder) -> None:
    """Each record() call must grow the timeline by one entry."""
    builder.record(0, [])
    assert len(builder.get_timeline()) == 1
    builder.record(1, [])
    assert len(builder.get_timeline()) == 2


def test_get_timeline_entry_is_timeline_entry(builder: TimelineBuilder) -> None:
    """Each element returned by get_timeline() must be a TimelineEntry."""
    builder.record(0, [_pred()])
    result = builder.get_timeline()
    assert isinstance(result[0], TimelineEntry)


def test_get_timeline_frame_index_preserved(builder: TimelineBuilder) -> None:
    """The recorded frame_index must be faithfully stored."""
    builder.record(17, [])
    assert builder.get_timeline()[0].frame_index == 17


def test_get_timeline_predictions_preserved(builder: TimelineBuilder) -> None:
    """The recorded predictions list must be faithfully stored."""
    preds = [_pred(1, "Safe", 0.91), _pred(2, "Critical", 0.78)]
    builder.record(0, preds)
    stored = builder.get_timeline()[0].predictions
    assert stored == preds


def test_record_empty_predictions_allowed(builder: TimelineBuilder) -> None:
    """record() with an empty predictions list must succeed without error."""
    try:
        builder.record(0, [])
    except Exception as exc:
        pytest.fail(f"record() raised unexpectedly with empty predictions: {exc}")


def test_get_timeline_returns_copy(builder: TimelineBuilder) -> None:
    """Mutating the returned list must not affect the builder's internal state."""
    builder.record(0, [])
    timeline = builder.get_timeline()
    timeline.append(TimelineEntry(frame_index=999, predictions=[]))
    # Internal state must still have only one entry
    assert len(builder.get_timeline()) == 1


def test_record_stores_copy_of_predictions(builder: TimelineBuilder) -> None:
    """Mutating the original predictions list after record() must not affect storage."""
    preds = [_pred(1, "Safe")]
    builder.record(0, preds)
    preds.append(_pred(2, "Critical"))   # mutate original
    stored = builder.get_timeline()[0].predictions
    assert len(stored) == 1  # stored list must be unchanged


def test_timeline_ordering_preserved(builder: TimelineBuilder) -> None:
    """Entries must be returned in the order they were recorded."""
    builder.record(0, [_pred(1, "Safe")])
    builder.record(1, [_pred(2, "Critical")])
    builder.record(2, [_pred(3, "Congesting")])
    timeline = builder.get_timeline()
    assert [e.frame_index for e in timeline] == [0, 1, 2]


# ---------------------------------------------------------------------------
# TimelineBuilder — reset
# ---------------------------------------------------------------------------


def test_reset_clears_state(builder: TimelineBuilder) -> None:
    """reset() must clear all recorded entries."""
    builder.record(0, [])
    builder.record(1, [])
    builder.reset()
    assert builder.get_timeline() == []


def test_reset_allows_reuse(builder: TimelineBuilder) -> None:
    """After reset(), the builder must accept new recordings correctly."""
    builder.record(0, [_pred()])
    builder.reset()
    builder.record(5, [_pred(1, "Critical")])
    timeline = builder.get_timeline()
    assert len(timeline) == 1
    assert timeline[0].frame_index == 5
