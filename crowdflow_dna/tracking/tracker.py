"""Multi-object tracking module for CrowdFlow DNA.

Implements a lightweight IoU-based tracker (inspired by SORT/ByteTrack)
using only numpy and scipy — libraries already present in the project
dependency stack. No additional third-party tracker library is introduced.

The tracker consumes BoundingBox detections from the Detection module and
produces a List[TrackItem] for the Pipeline orchestrator, which is the
stable handshake contract consumed by downstream modules.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from crowdflow_dna import config
from crowdflow_dna.errors import ModelInferenceError
from crowdflow_dna.schemas import TrackItem

logger = logging.getLogger(__name__)

# Type alias imported from Detection module for clarity
BoundingBox = Tuple[float, float, float, float, float, int]


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two bounding boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def _iou_matrix(
    tracks: List[np.ndarray], detections: List[np.ndarray]
) -> np.ndarray:
    """Build an (M, N) IoU cost matrix between M tracks and N detections."""
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            matrix[i, j] = _iou(t, d)
    return matrix


class _Track:
    """Internal state for a single tracked pedestrian."""

    _id_counter: int = 0

    def __init__(self, bbox: np.ndarray) -> None:
        _Track._id_counter += 1
        self.track_id: int = _Track._id_counter
        self.bbox: np.ndarray = bbox.copy()
        self.prev_centroid: Optional[Tuple[float, float]] = None

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the global ID counter (for testing only)."""
        cls._id_counter = 0

    def centroid(self) -> Tuple[float, float]:
        """Compute centroid from current bbox."""
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return float(cx), float(cy)


class ByteTracker:
    """Lightweight IoU-based multi-object tracker.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to
    associate detection bounding boxes across frames via IoU matching —
    the same association strategy as SORT and ByteTrack.

    Built exclusively on numpy and scipy, which are already present in the
    project dependency stack. No new third-party tracker library is introduced.

    Velocity ``(vx, vy)`` is computed by differencing each track's centroid
    between the current and previous frame. Tracks on their first appearance
    receive velocity ``(0.0, 0.0)``. Disappeared tracks are immediately
    removed from internal state so memory does not grow indefinitely.

    Example usage::

        tracker = ByteTracker()
        for frame_detections in per_frame_detections:
            track_items: List[TrackItem] = tracker.update(frame_detections)
    """

    def __init__(self) -> None:
        """Initialise the tracker.

        Reads ``IOU_THRESHOLD`` from ``config`` to control the minimum IoU
        required for a detection to be matched to an existing track.
        """
        self._iou_threshold: float = float(config.IOU_THRESHOLD)
        self._active_tracks: Dict[int, _Track] = {}
        logger.info(
            "ByteTracker initialised (iou_threshold=%.2f)", self._iou_threshold
        )

    def update(self, detections: List[BoundingBox]) -> List[TrackItem]:
        """Update the tracker with detections for the current frame.

        Args:
            detections: List of ``BoundingBox`` tuples
                ``(x1, y1, x2, y2, confidence, class_id)`` as produced by
                ``Yolov8Detector.detect()``. May be an empty list.

        Returns:
            A list of :class:`~crowdflow_dna.schemas.TrackItem` dataclasses,
            one per active track in this frame. Returns an empty list when
            no detections are provided or all detections fail to match.
            Tracks that disappear in this frame are removed from internal
            state immediately; they will receive a new ``track_id`` if they
            reappear in a later frame.

        Raises:
            ModelInferenceError: If the tracking association step raises an
                unexpected exception.
        """
        try:
            return self._update(detections)
        except ModelInferenceError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                f"ByteTracker.update() failed unexpectedly: {exc}"
            ) from exc

    def _update(self, detections: List[BoundingBox]) -> List[TrackItem]:
        """Internal update logic, separated for clean exception wrapping."""
        det_boxes = [
            np.array([d[0], d[1], d[2], d[3]], dtype=np.float64)
            for d in detections
        ]

        if not det_boxes:
            # No detections: all active tracks disappear.
            self._active_tracks.clear()
            return []

        if not self._active_tracks:
            # No existing tracks: create a new track for every detection.
            self._create_new_tracks(det_boxes)
            return self._build_track_items()

        track_list = list(self._active_tracks.values())
        track_boxes = [t.bbox for t in track_list]

        # Hungarian assignment on the IoU cost matrix.
        iou_mat = _iou_matrix(track_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)  # maximise IoU

        matched_track_ids = set()
        matched_det_idxs = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= self._iou_threshold:
                matched_track_ids.add(track_list[r].track_id)
                matched_det_idxs.add(c)
                track_list[r].bbox = det_boxes[c].copy()

        # Remove tracks that were not matched (they disappeared this frame).
        # Explicit cleanup prevents _active_tracks from growing indefinitely.
        disappeared_ids = [
            t.track_id
            for t in track_list
            if t.track_id not in matched_track_ids
        ]
        for tid in disappeared_ids:
            del self._active_tracks[tid]
            logger.debug("Track %d lost — removed from state", tid)

        # Create new tracks for unmatched detections.
        unmatched_dets = [
            det_boxes[i]
            for i in range(len(det_boxes))
            if i not in matched_det_idxs
        ]
        self._create_new_tracks(unmatched_dets)

        # _build_track_items is called exactly once per update() call.
        # Calling it more than once would update prev_centroid prematurely,
        # causing velocity to read as (0, 0) on subsequent frames.
        return self._build_track_items()

    def _create_new_tracks(self, det_boxes: List[np.ndarray]) -> None:
        """Register new Track objects for each unmatched detection box.

        Does NOT call _build_track_items; that must be done exactly once
        by the caller (_update) after all state mutations are complete.
        """
        for box in det_boxes:
            track = _Track(box)
            self._active_tracks[track.track_id] = track

    def _build_track_items(self) -> List[TrackItem]:
        """Convert all active _Track objects to TrackItem dataclasses.

        Velocity is computed by differencing the current centroid from the
        centroid stored at the end of the previous frame. Tracks appearing
        for the first time receive velocity (0.0, 0.0).

        After velocities are computed, ``prev_centroid`` is updated on every
        active track so the next call has fresh reference values.
        """
        items: List[TrackItem] = []

        for track in self._active_tracks.values():
            cx, cy = track.centroid()

            if track.prev_centroid is None:
                vx, vy = 0.0, 0.0
            else:
                vx = cx - track.prev_centroid[0]
                vy = cy - track.prev_centroid[1]

            # Update stored centroid for next frame's velocity computation.
            track.prev_centroid = (cx, cy)

            items.append(
                TrackItem(
                    track_id=track.track_id,
                    bbox=(
                        float(track.bbox[0]),
                        float(track.bbox[1]),
                        float(track.bbox[2]),
                        float(track.bbox[3]),
                    ),
                    centroid=(cx, cy),
                    velocity=(float(vx), float(vy)),
                )
            )

        logger.debug("ByteTracker: %d active track(s) this frame", len(items))
        return items
