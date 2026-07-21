"""Pipeline orchestration module for CrowdFlow DNA.

Wires all completed pipeline stages end-to-end:

    VideoIngestor → Yolov8Detector → ByteTracker
        → GraphBuilder → model_fn (optional)
        → FrameAnnotator → TimelineBuilder

The pipeline is the single entry point for processing a video file.
All component logic remains in the respective modules; this module
contains only orchestration and data-routing code.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from crowdflow_dna import config
from crowdflow_dna.detection.detector import Yolov8Detector
from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.graph.graph_builder import GraphBuilder
from crowdflow_dna.ingestion.video_loader import VideoIngestor
from crowdflow_dna.rendering.timeline import TimelineBuilder, TimelineEntry
from crowdflow_dna.rendering.video_renderer import FrameAnnotator
from crowdflow_dna.schemas import RiskPrediction, TrackItem
from crowdflow_dna.tracking.tracker import ByteTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured output of a single ``CrowdFlowPipeline.run()`` call.

    Attributes:
        annotated_frames: Ordered list of BGR frames with bounding boxes,
            track IDs, and risk labels drawn on them. One entry per
            sampled frame (sampled at ``config.FRAME_SAMPLE_RATE``).
        timeline: Per-frame risk accumulation, one :class:`TimelineEntry`
            per sampled frame. Consumed by the Gradio UI (Phase 8).
        metadata: Stream-level metadata dict produced by
            :class:`~crowdflow_dna.ingestion.video_loader.VideoIngestor`,
            containing keys ``fps``, ``width``, ``height``, ``frame_count``,
            ``duration_seconds``, and ``sample_rate``.
    """

    annotated_frames: List[np.ndarray] = field(default_factory=list)
    timeline: List[TimelineEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CrowdFlowPipeline:
    """End-to-end orchestrator for the CrowdFlow DNA vision pipeline.

    Connects all completed modules in sequence, routing data between
    stages according to their published interfaces. No module logic is
    duplicated here; this class is purely orchestration.

    **Model injection**

    The GNN/risk-classification model (Phase 4/5, Piyush) is injected as
    an optional callable. When ``model_fn`` is ``None`` (the default), the
    pipeline operates in *dummy mode*: no risk predictions are generated and
    all bounding boxes are rendered in neutral grey. This satisfies the
    Phase 7/8 Definition of Done — *"Gradio UI built using dummy labels"* —
    and allows the full pipeline to run without trained model weights.

    When Piyush's model is available (Phase 9), it is wired in by passing
    a callable that accepts a ``torch_geometric.data.Data`` graph and returns
    a ``List[RiskPrediction]``. No changes to this class are required.

    **GraphBuilder integration**

    ``GraphBuilder.build()`` requires positions normalised to ``[0, 1]``.
    Centroids from ``TrackItem`` are normalised by frame width and height.
    The proximity radius from ``config.PROXIMITY_RADIUS`` (pixels) is
    similarly normalised before passing to ``GraphBuilder.__init__``.
    If the normalised radius would exceed ``1.0`` (e.g. on tiny frames),
    it is clamped to ``1.0`` with a warning.

    Graph construction is skipped when the track list is empty; the model
    callable is never invoked in that case.

    Example usage::

        pipeline = CrowdFlowPipeline()
        result = pipeline.run("crowd_video.mp4")
        # result.annotated_frames — list of BGR frames
        # result.timeline         — List[TimelineEntry]
        # result.metadata         — fps, width, height, …
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[Any], List[RiskPrediction]]] = None,
    ) -> None:
        """Initialise all pipeline components.

        Args:
            model_fn: An optional callable that accepts a
                ``torch_geometric.data.Data`` graph for a single frame and
                returns a ``List[RiskPrediction]``. When ``None``, the
                pipeline runs in dummy mode with no risk predictions.
        """
        self._ingestor = VideoIngestor()
        self._detector = Yolov8Detector()
        self._tracker = ByteTracker()
        self._annotator = FrameAnnotator()
        self._timeline = TimelineBuilder()
        self._model_fn = model_fn

        logger.info(
            "CrowdFlowPipeline initialised (model_fn=%s)",
            "provided" if model_fn is not None else "None — dummy mode",
        )

    def run(self, video_path: str) -> PipelineResult:
        """Process a video file end-to-end.

        Loads the video, runs detection and tracking on each sampled frame,
        optionally builds a graph and runs the risk model, annotates frames,
        and accumulates a risk timeline.

        Args:
            video_path: Absolute or relative path to an MP4 or AVI file.

        Returns:
            A :class:`PipelineResult` containing annotated frames, the risk
            timeline, and stream metadata.

        Raises:
            InvalidVideoFormatError: If the video format is unsupported.
            UploadSizeExceededError: If the file exceeds size/duration limits.
            VideoCorruptionError: If the file cannot be opened or read.
            ModelInferenceError: If detection or tracking raises internally.
            CrowdFlowError: For any other unexpected failure inside the loop.
        """
        logger.info("Pipeline starting for: %s", video_path)

        # ----------------------------------------------------------------
        # Stage 1: Ingestion
        # ----------------------------------------------------------------
        frames, metadata = self._ingestor.load(video_path)
        logger.info(
            "Ingestion complete: %d frames, %.1f fps, %dx%d",
            len(frames),
            metadata["fps"],
            metadata["width"],
            metadata["height"],
        )

        if not frames:
            logger.warning("No frames sampled from %s — returning empty result.", video_path)
            return PipelineResult(metadata=metadata)

        # ----------------------------------------------------------------
        # Stage 2: Build GraphBuilder with normalised proximity radius
        # Verified from graph_builder.py:
        #   - proximity_radius must be in (0, 1]
        #   - config.PROXIMITY_RADIUS is in pixels → must normalise
        # ----------------------------------------------------------------
        width: int = metadata["width"]
        height: int = metadata["height"]
        min_dim = min(width, height)

        raw_radius = float(config.PROXIMITY_RADIUS)
        norm_radius = raw_radius / min_dim if min_dim > 0 else 1.0

        if norm_radius > 1.0:
            logger.warning(
                "Normalised proximity radius %.4f > 1.0 (frame %dx%d, "
                "PROXIMITY_RADIUS=%.1f). Clamping to 1.0.",
                norm_radius,
                width,
                height,
                raw_radius,
            )
            norm_radius = 1.0

        graph_builder = GraphBuilder(proximity_radius=norm_radius)

        # ----------------------------------------------------------------
        # Stage 3: Reset timeline for this run
        # ----------------------------------------------------------------
        self._timeline.reset()
        annotated_frames: List[np.ndarray] = []

        # ----------------------------------------------------------------
        # Stage 4: Per-frame loop
        # ----------------------------------------------------------------
        for frame_index, frame in enumerate(frames):
            try:
                annotated, predictions = self._process_frame(
                    frame_index, frame, graph_builder, width, height
                )
                annotated_frames.append(annotated)
                self._timeline.record(frame_index, predictions)

            except CrowdFlowError:
                raise
            except Exception as exc:
                raise CrowdFlowError(
                    f"Unexpected error on frame {frame_index}: {exc}"
                ) from exc

        result = PipelineResult(
            annotated_frames=annotated_frames,
            timeline=self._timeline.get_timeline(),
            metadata=metadata,
        )
        logger.info(
            "Pipeline complete: %d annotated frames produced.", len(annotated_frames)
        )
        return result

    def _process_frame(
        self,
        frame_index: int,
        frame: np.ndarray,
        graph_builder: GraphBuilder,
        width: int,
        height: int,
    ) -> tuple:
        """Process a single frame through detection, tracking, graph, and rendering.

        Args:
            frame_index: Zero-based index of this frame in the sampled sequence.
            frame: BGR NumPy array from the ingestion stage.
            graph_builder: Pre-initialised GraphBuilder for this video.
            width: Frame width in pixels (used for centroid normalisation).
            height: Frame height in pixels (used for centroid normalisation).

        Returns:
            Tuple of ``(annotated_frame, predictions)`` where
            ``annotated_frame`` is a BGR ``np.ndarray`` and ``predictions``
            is a ``List[RiskPrediction]`` (may be empty).
        """
        # Stage 4a — Detection
        detections = self._detector.detect(frame)

        # Stage 4b — Tracking
        tracks: List[TrackItem] = self._tracker.update(detections)

        # Stage 4c — Graph + Model (only when tracks exist and model provided)
        predictions: List[RiskPrediction] = []
        if tracks and self._model_fn is not None:
            positions, velocities = self._extract_arrays(tracks, width, height)
            try:
                graph = graph_builder.build(positions, velocities)
                predictions = self._model_fn(graph)
            except ImportError:
                # torch_geometric not installed — skip graph/model silently
                logger.warning(
                    "torch_geometric not available; skipping graph/model on frame %d.",
                    frame_index,
                )
            except Exception as exc:
                raise CrowdFlowError(
                    f"Graph/model step failed on frame {frame_index}: {exc}"
                ) from exc

        # Stage 4d — Annotation
        annotated = self._annotator.annotate(frame, tracks, predictions)

        logger.debug(
            "Frame %d: %d detection(s), %d track(s), %d prediction(s)",
            frame_index,
            len(detections),
            len(tracks),
            len(predictions),
        )
        return annotated, predictions

    @staticmethod
    def _extract_arrays(
        tracks: List[TrackItem],
        width: int,
        height: int,
    ) -> tuple:
        """Convert TrackItem centroids and velocities to GraphBuilder arrays.

        Verified against graph_builder.py:
        - positions: shape (N, 2), values normalised to [0, 1]
          by dividing cx by frame width and cy by frame height.
        - velocities: shape (N, 2), pixel-per-frame deltas, float32-compatible.
          GraphBuilder does NOT require normalised velocities — it accepts
          any float32-compatible values and uses them directly for speed/edge
          feature computation.

        Args:
            tracks: Non-empty list of TrackItem objects for a single frame.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            Tuple ``(positions, velocities)`` where both are ``np.ndarray``
            of shape ``(N, 2)`` and dtype ``float32``.
        """
        positions = np.array(
            [[t.centroid[0] / width, t.centroid[1] / height] for t in tracks],
            dtype=np.float32,
        )
        velocities = np.array(
            [[t.velocity[0], t.velocity[1]] for t in tracks],
            dtype=np.float32,
        )
        return positions, velocities
