"""Risk timeline module for CrowdFlow DNA.

Accumulates per-frame risk predictions across the lifetime of a video
and exposes them as a structured list consumed by the Gradio UI.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from crowdflow_dna.schemas import RiskPrediction

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single frame's risk state in the processed video.

    Attributes:
        frame_index: Zero-based index of the video frame.
        predictions: Risk predictions for this frame. May be an empty list
            when the model has not yet produced output (e.g., dummy runs).
    """

    frame_index: int
    predictions: List[RiskPrediction] = field(default_factory=list)


class TimelineBuilder:
    """Accumulates per-frame risk predictions into a structured timeline.

    Maintains an ordered list of :class:`TimelineEntry` objects — one per
    recorded frame. The timeline is consumed by the Gradio UI (Phase 8) to
    render a risk chart alongside the annotated video.

    Call :meth:`reset` between videos to clear state; the builder is
    intentionally designed to be reused across multiple pipeline runs.

    Example usage::

        builder = TimelineBuilder()
        for idx, preds in enumerate(per_frame_predictions):
            builder.record(idx, preds)
        timeline = builder.get_timeline()
    """

    def __init__(self) -> None:
        """Initialise an empty timeline."""
        self._entries: List[TimelineEntry] = []

    def record(
        self,
        frame_index: int,
        predictions: List[RiskPrediction],
    ) -> None:
        """Record the risk predictions for a single frame.

        Args:
            frame_index: Zero-based index of the video frame being recorded.
            predictions: Risk predictions from the model for this frame.
                An empty list is valid (e.g., for dummy/no-model runs or
                frames where no pedestrians were detected).
        """
        entry = TimelineEntry(frame_index=frame_index, predictions=list(predictions))
        self._entries.append(entry)
        logger.debug(
            "TimelineBuilder.record(): frame %d — %d prediction(s)",
            frame_index,
            len(predictions),
        )

    def get_timeline(self) -> List[TimelineEntry]:
        """Return a copy of the accumulated timeline.

        Returns:
            A new list of :class:`TimelineEntry` objects in recording order.
            Mutating the returned list does not affect internal state.
        """
        return list(self._entries)

    def reset(self) -> None:
        """Clear all recorded entries.

        Call this between videos to prevent state from bleeding across
        pipeline runs in a long-running service.
        """
        self._entries.clear()
        logger.debug("TimelineBuilder.reset(): timeline cleared")
