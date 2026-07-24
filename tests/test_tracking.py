"""Unit tests for ByteTracker (crowdflow_dna/tracking/tracker.py).

Tests are fully deterministic and require no real video frames or GPU.
The _Track ID counter is reset before each test to ensure isolated, stable IDs.
"""

import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna.errors import ModelInferenceError
from crowdflow_dna.schemas import TrackItem
from crowdflow_dna.tracking.tracker import ByteTracker, _Track

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A BoundingBox tuple: (x1, y1, x2, y2, confidence, class_id)
def _box(x1: float, y1: float, x2: float, y2: float) -> tuple:
    return (x1, y1, x2, y2, 0.9, 0)


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset the global _Track ID counter before every test for isolation."""
    _Track.reset_id_counter()
    yield
    _Track.reset_id_counter()


@pytest.fixture()
def tracker() -> ByteTracker:
    return ByteTracker()


# ---------------------------------------------------------------------------
# Basic return type and empty-input tests
# ---------------------------------------------------------------------------


def test_update_always_returns_list(tracker: ByteTracker) -> None:
    """update() must always return a list."""
    result = tracker.update([])
    assert isinstance(result, list)


def test_empty_detections_returns_empty_list(tracker: ByteTracker) -> None:
    """No detections → empty list, no exception."""
    assert tracker.update([]) == []


def test_empty_detections_no_exception(tracker: ByteTracker) -> None:
    """Empty detection list must never raise."""
    try:
        tracker.update([])
    except Exception as exc:
        pytest.fail(f"update([]) raised unexpectedly: {exc}")


def test_single_detection_returns_one_track(tracker: ByteTracker) -> None:
    """One detection produces exactly one TrackItem."""
    result = tracker.update([_box(10, 20, 100, 200)])
    assert len(result) == 1


def test_multiple_detections_all_tracked(tracker: ByteTracker) -> None:
    """Each detection produces one TrackItem when no overlap exists."""
    dets = [_box(0, 0, 50, 50), _box(200, 200, 300, 300), _box(400, 0, 500, 100)]
    result = tracker.update(dets)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# TrackItem contract tests
# ---------------------------------------------------------------------------


def test_trackitem_has_all_fields(tracker: ByteTracker) -> None:
    """Every returned object must be a TrackItem with all four fields."""
    result = tracker.update([_box(10, 20, 100, 200)])
    item = result[0]
    assert isinstance(item, TrackItem)
    field_names = {f.name for f in fields(TrackItem)}
    assert field_names == {"track_id", "bbox", "centroid", "velocity"}


def test_trackitem_field_types(tracker: ByteTracker) -> None:
    """TrackItem fields must have the contracted types."""
    result = tracker.update([_box(10.0, 20.0, 100.0, 200.0)])
    item = result[0]
    assert isinstance(item.track_id, int)
    assert isinstance(item.bbox, tuple) and len(item.bbox) == 4
    assert isinstance(item.centroid, tuple) and len(item.centroid) == 2
    assert isinstance(item.velocity, tuple) and len(item.velocity) == 2
    assert all(isinstance(v, float) for v in item.bbox)
    assert all(isinstance(v, float) for v in item.centroid)
    assert all(isinstance(v, float) for v in item.velocity)


def test_bbox_matches_detection(tracker: ByteTracker) -> None:
    """TrackItem.bbox must equal the input detection's (x1,y1,x2,y2)."""
    result = tracker.update([_box(15.5, 30.0, 120.0, 250.5)])
    x1, y1, x2, y2 = result[0].bbox
    assert x1 == pytest.approx(15.5)
    assert y1 == pytest.approx(30.0)
    assert x2 == pytest.approx(120.0)
    assert y2 == pytest.approx(250.5)


def test_centroid_computed_correctly(tracker: ByteTracker) -> None:
    """Centroid must be (cx, cy) = ((x1+x2)/2, (y1+y2)/2)."""
    result = tracker.update([_box(0.0, 0.0, 100.0, 200.0)])
    cx, cy = result[0].centroid
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Velocity tests
# ---------------------------------------------------------------------------


def test_velocity_zero_on_first_appearance(tracker: ByteTracker) -> None:
    """A track's velocity is (0.0, 0.0) on its first frame."""
    result = tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    vx, vy = result[0].velocity
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(0.0)


def test_velocity_computed_from_centroid_diff(tracker: ByteTracker) -> None:
    """After two frames, velocity must equal the centroid delta."""
    # Frame 1: centroid at (50, 50)
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    # Frame 2: same track shifts right by 20px → centroid at (70, 50)
    result = tracker.update([_box(20.0, 0.0, 120.0, 100.0)])
    vx, vy = result[0].velocity
    assert vx == pytest.approx(20.0)
    assert vy == pytest.approx(0.0)


def test_velocity_updates_correctly_over_multiple_frames(tracker: ByteTracker) -> None:
    """Velocity must reflect the most recent centroid delta, not cumulative."""
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])   # centroid (50, 50)
    tracker.update([_box(20.0, 0.0, 120.0, 100.0)])  # centroid (70, 50) → v=(20,0)
    result = tracker.update([_box(20.0, 10.0, 120.0, 110.0)])  # centroid (70, 60) → v=(0,10)
    vx, vy = result[0].velocity
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Track ID persistence tests
# ---------------------------------------------------------------------------


def test_track_id_persists_across_frames(tracker: ByteTracker) -> None:
    """The same track ID must be assigned to a matched detection next frame."""
    r1 = tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    first_id = r1[0].track_id
    r2 = tracker.update([_box(5.0, 5.0, 105.0, 105.0)])  # high IoU — same track
    assert r2[0].track_id == first_id


def test_new_detection_gets_new_id(tracker: ByteTracker) -> None:
    """A non-overlapping detection in a second frame gets a new track ID."""
    r1 = tracker.update([_box(0.0, 0.0, 10.0, 10.0)])
    first_id = r1[0].track_id
    r2 = tracker.update([_box(500.0, 500.0, 600.0, 600.0)])  # no IoU — new track
    assert r2[0].track_id != first_id


# ---------------------------------------------------------------------------
# Track disappearance and state cleanup tests
# ---------------------------------------------------------------------------


def test_disappeared_track_removed_from_state(tracker: ByteTracker) -> None:
    """When a track is not matched, it must be removed from internal state."""
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    # Pass a detection with no IoU overlap → original track disappears
    tracker.update([_box(500.0, 500.0, 600.0, 600.0)])
    assert 1 not in tracker._active_tracks  # track_id 1 must be gone


def test_state_does_not_grow_after_disappearance(tracker: ByteTracker) -> None:
    """Internal state size must not grow beyond the number of active tracks."""
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    assert len(tracker._active_tracks) == 1
    # Non-overlapping → original disappears, new one appears
    tracker.update([_box(500.0, 500.0, 600.0, 600.0)])
    assert len(tracker._active_tracks) == 1  # not 2


def test_empty_frame_clears_all_state(tracker: ByteTracker) -> None:
    """An empty detection frame must clear all active tracks immediately."""
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    tracker.update([])  # all tracks disappear
    assert len(tracker._active_tracks) == 0


def test_velocity_zero_for_reappearing_track(tracker: ByteTracker) -> None:
    """A track that disappears and reappears must get a fresh track ID and
    velocity (0.0, 0.0) on its first re-detection, since state was cleared.

    This validates that _prev_centroids / velocity state is not leaked
    across disappearance events — reappearing pedestrians are treated as
    new tracks, not as continuations of lost ones.
    """
    # Frame 1: track appears at centroid (50, 50)
    r1 = tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    original_id = r1[0].track_id

    # Frame 2: track builds velocity → centroid (70, 50)
    tracker.update([_box(20.0, 0.0, 120.0, 100.0)])

    # Frame 3: track disappears (no matching detection)
    tracker.update([])

    # Frame 4: a detection reappears at roughly the same location
    r4 = tracker.update([_box(20.0, 0.0, 120.0, 100.0)])

    # Must receive a brand new ID (old one was removed when it disappeared)
    new_id = r4[0].track_id
    assert new_id != original_id, "Reappearing track must get a new ID"

    # Velocity must be (0.0, 0.0) because this is a fresh track, not a resumed one
    vx, vy = r4[0].velocity
    assert vx == pytest.approx(0.0), "Reappearing track must have vx=0"
    assert vy == pytest.approx(0.0), "Reappearing track must have vy=0"


# ---------------------------------------------------------------------------
# IoU matching tests
# ---------------------------------------------------------------------------


def test_high_iou_detection_matched_to_existing_track(tracker: ByteTracker) -> None:
    """A detection with IoU >= threshold is matched, not treated as new."""
    r1 = tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
    first_id = r1[0].track_id
    # Slight shift — high IoU with original
    r2 = tracker.update([_box(2.0, 2.0, 102.0, 102.0)])
    assert r2[0].track_id == first_id
    assert len(tracker._active_tracks) == 1


def test_low_iou_detection_creates_new_track(tracker: ByteTracker) -> None:
    """A detection with IoU < threshold creates a new track."""
    tracker.update([_box(0.0, 0.0, 10.0, 10.0)])
    r2 = tracker.update([_box(900.0, 900.0, 950.0, 950.0)])
    assert len(r2) == 1  # old track gone, new one added
    assert len(tracker._active_tracks) == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_tracking_failure_raises_model_inference_error(tracker: ByteTracker) -> None:
    """If the internal update raises, ModelInferenceError must propagate."""
    with patch(
        "crowdflow_dna.tracking.tracker.linear_sum_assignment",
        side_effect=RuntimeError("matrix error"),
    ):
        tracker.update([_box(0.0, 0.0, 100.0, 100.0)])  # prime with a track
        with pytest.raises(ModelInferenceError, match="ByteTracker"):
            tracker.update([_box(5.0, 5.0, 105.0, 105.0)])


def test_tracking_error_chains_original_exception(tracker: ByteTracker) -> None:
    """ModelInferenceError must chain the original exception via __cause__."""
    original = ValueError("unexpected numpy error")
    with patch(
        "crowdflow_dna.tracking.tracker.linear_sum_assignment",
        side_effect=original,
    ):
        tracker.update([_box(0.0, 0.0, 100.0, 100.0)])
        with pytest.raises(ModelInferenceError) as exc_info:
            tracker.update([_box(5.0, 5.0, 105.0, 105.0)])
        assert exc_info.value.__cause__ is original
