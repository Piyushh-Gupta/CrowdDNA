"""Frame annotation module for CrowdFlow DNA.

Draws bounding boxes, track IDs, and colour-coded risk labels onto
individual video frames. Sits between the Tracking module and the
Gradio UI in the pipe-and-filter pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.schemas import RiskPrediction, TrackItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — BGR format used by OpenCV
# ---------------------------------------------------------------------------

# Maps RiskPrediction.label → BGR box colour
_RISK_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Safe":       (0, 200,   0),   # green
    "Congesting": (0, 165, 255),   # orange
    "Critical":   (0,   0, 220),   # red
}

# Colour used when no risk prediction is available for a track
_DEFAULT_COLOUR: Tuple[int, int, int] = (180, 180, 180)   # neutral grey

# Drawing constants
_BOX_THICKNESS: int = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE: float = 0.55
_FONT_THICKNESS: int = 1
_LABEL_PAD: int = 4   # pixels above bbox for text


class FrameAnnotator:
    """Annotates a single BGR video frame with tracking and risk data.

    Draws a coloured bounding box and text label for every ``TrackItem``.
    Box colour is determined by the ``RiskPrediction`` whose ``region_id``
    matches the track's ``track_id``; tracks with no matching prediction
    are drawn in neutral grey.

    This class is stateless — every call to :meth:`annotate` is independent.
    The input frame is never mutated; a copy is always returned.

    Example usage::

        annotator = FrameAnnotator()
        annotated = annotator.annotate(frame, tracks, predictions)
    """

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[TrackItem],
        predictions: List[RiskPrediction],
    ) -> np.ndarray:
        """Annotate a single frame with bounding boxes and risk labels.

        Args:
            frame: BGR image as a NumPy ndarray (H, W, 3), as produced by
                ``VideoIngestor.load()``. The array is not mutated.
            tracks: Active tracks for this frame from ``ByteTracker.update()``.
                May be an empty list.
            predictions: Risk predictions for this frame from the model.
                May be an empty list (e.g., during dummy / no-model runs).

        Returns:
            A new ``np.ndarray`` (same shape and dtype as ``frame``) with
            bounding boxes, track IDs, and risk labels drawn on it.

        Raises:
            CrowdFlowError: If an OpenCV drawing operation fails unexpectedly.
        """
        # Always work on a copy — never mutate the caller's frame.
        canvas: np.ndarray = frame.copy()

        if not tracks:
            return canvas

        # Build a lookup from track_id → risk label for O(1) matching.
        risk_map: Dict[int, RiskPrediction] = {
            p.region_id: p for p in predictions
        }

        try:
            for track in tracks:
                pred: Optional[RiskPrediction] = risk_map.get(track.track_id)
                colour = (
                    _RISK_COLOURS.get(pred.label, _DEFAULT_COLOUR)
                    if pred is not None
                    else _DEFAULT_COLOUR
                )
                self._draw_track(canvas, track, pred, colour)
        except Exception as exc:
            raise CrowdFlowError(
                f"FrameAnnotator failed during OpenCV drawing: {exc}"
            ) from exc

        logger.debug(
            "annotate(): drew %d track(s) on frame", len(tracks)
        )
        return canvas

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_track(
        canvas: np.ndarray,
        track: TrackItem,
        pred: Optional[RiskPrediction],
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw a single track's bbox, ID label, and optional risk label.

        Args:
            canvas: BGR frame to draw onto (modified in-place).
            track: The track whose bbox and ID are drawn.
            pred: The matching risk prediction, or ``None``.
            colour: BGR colour tuple for the bounding box and text.
        """
        x1, y1, x2, y2 = (int(v) for v in track.bbox)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

        # Build the overlay text: "ID:N  Label conf" or just "ID:N"
        id_text = f"ID:{track.track_id}"
        if pred is not None:
            risk_text = f"{pred.label} {pred.confidence:.2f}"
            label = f"{id_text}  {risk_text}"
        else:
            label = id_text

        # Position text above the box; clamp to frame top edge.
        text_y = max(y1 - _LABEL_PAD, _LABEL_PAD + 10)
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            _FONT,
            _FONT_SCALE,
            colour,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
