"""Pipeline orchestration module for CrowdFlow DNA.

Wires all completed pipeline stages end-to-end:

    VideoIngestor → Yolov8Detector → ByteTracker
        → GraphBuilder → SequenceBuffer → InferenceRuntime (optional)
        → FrameAnnotator → TimelineBuilder

The pipeline is the single entry point for processing a video file.
All component logic remains in the respective modules; this module
contains only orchestration and data-routing code.

When ``model_path`` is provided, the pipeline runs in **inference mode**:
``SequenceBuffer`` accumulates ``WINDOW_SIZE`` per-frame graphs, then
``InferenceRuntime.predict()`` produces an ``InferenceResult`` which is
translated into a ``List[RiskPrediction]`` for the rendering layer.

When ``model_path`` is ``None``, the pipeline operates in **dummy mode**:
no risk predictions are generated and all bounding boxes are rendered in
neutral grey. This satisfies the Phase 7/8 Definition of Done.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

from crowdflow_dna import config
from crowdflow_dna.detection.detector import Yolov8Detector
from crowdflow_dna.errors import CrowdFlowError, ModelInferenceError
from crowdflow_dna.graph.graph_builder import GraphBuilder
from crowdflow_dna.inference import (
    InferenceExecutionError,
    InferenceResult,
    InferenceRuntime,
    SequenceBuffer,
)
from crowdflow_dna.inference.sequence_buffer import (
    _EDGE_FEATURE_DIM,
    _NODE_FEATURE_DIM,
)
from crowdflow_dna.ingestion.video_loader import VideoIngestor
from crowdflow_dna.rendering.timeline import TimelineBuilder, TimelineEntry
from crowdflow_dna.rendering.video_renderer import FrameAnnotator
from crowdflow_dna.schemas import RiskPrediction, TrackItem
from crowdflow_dna.tracking.tracker import ByteTracker

logger = logging.getLogger(__name__)

# Canonical class index → risk label mapping (must match model training).
_CLASS_NAMES: List[str] = ["Safe", "Congesting", "Critical"]

# Sequence buffer window size (frames per inference call). See INTEGRATION_CONTRACT.md §6.
_WINDOW_SIZE: int = 10


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
    stages according to the published interfaces defined in
    ``docs/INTEGRATION_CONTRACT.md``.

    **Inference mode**

    Pass ``model_path`` to load a TorchScript (``.pt``) or ONNX
    (``.onnx``) model exported by ``ModelExporter``. The pipeline will
    buffer ``WINDOW_SIZE`` consecutive frames of graph data and invoke
    ``InferenceRuntime.predict()`` once the buffer is full. Until the buffer
    warms up (first ``WINDOW_SIZE`` frames), frames are rendered without
    risk predictions.

    **Dummy mode**

    When ``model_path`` is ``None`` (the default), the pipeline operates
    in dummy mode: no risk predictions are generated and all bounding boxes
    are rendered in neutral grey.

    Example usage::

        pipeline = CrowdFlowPipeline(model_path="exports/deployment.pt")
        result = pipeline.run("crowd_video.mp4")
        # result.annotated_frames — list of BGR frames
        # result.timeline         — List[TimelineEntry]
        # result.metadata         — fps, width, height, …
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_version: Optional[str] = None,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        """Initialise all pipeline components.

        Args:
            model_path: Path to a ``.pt`` or ``.onnx`` deployment model
                exported by ``ModelExporter``. When ``None``, the pipeline
                runs in dummy mode with no risk predictions.
            model_version: Optional version string forwarded to
                ``InferenceRuntime.load_model()`` for provenance tracking.
            window_size: Number of consecutive frames per inference call.
                Must be ≥ 1. Defaults to ``_WINDOW_SIZE`` (10).

        Raises:
            ModelNotFoundError: If ``model_path`` is provided but the file
                does not exist.
            UnsupportedModelFormatError: If the file extension is not
                ``.pt`` or ``.onnx``.
            InferenceExecutionError: If the model file is corrupt or
                incompatible.
        """
        self._ingestor = VideoIngestor()
        self._detector = Yolov8Detector()
        self._tracker = ByteTracker()
        self._annotator = FrameAnnotator()
        self._timeline = TimelineBuilder()

        self._runtime: Optional[InferenceRuntime] = None
        self._seq_buffer: Optional[SequenceBuffer] = None

        if model_path is not None:
            self._runtime = InferenceRuntime()
            self._runtime.load_model(model_path, version=model_version)
            self._seq_buffer = SequenceBuffer(window_size=window_size)
            logger.info(
                "CrowdFlowPipeline initialised in inference mode: %s (window=%d)",
                model_path,
                window_size,
            )
        else:
            logger.info("CrowdFlowPipeline initialised in dummy mode (no model).")

    def run(self, video_path: str) -> PipelineResult:
        """Process a video file end-to-end.

        Loads the video, runs detection and tracking on each sampled frame,
        optionally buffers graphs and runs the risk model, annotates frames,
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
            ModelInferenceError: If the runtime raises during inference.
            CrowdFlowError: For any other unexpected failure inside the loop.
        """
        logger.info("Entering CrowdFlowPipeline.run() for video: %s", video_path)

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

        thread_id = __import__('threading').get_ident()
        if not frames:
            logger.warning(
                "[Thread %s] No frames sampled from %s — returning empty result.", thread_id, video_path
            )
            logger.info("[Thread %s] Exiting CrowdFlowPipeline.run() early: no frames", thread_id)
            return PipelineResult(metadata=metadata)

        # ----------------------------------------------------------------
        # Stage 2: Build GraphBuilder with normalised proximity radius
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
        # Stage 3: Reset stateful components for this run
        # ----------------------------------------------------------------
        self._timeline.reset()
        if self._seq_buffer is not None:
            self._seq_buffer.reset()

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
                logger.info("[Thread %s] run() exception handler: CrowdFlowError on frame %d", thread_id, frame_index)
                raise
            except Exception as exc:
                logger.info("[Thread %s] run() exception handler: Unexpected error on frame %d", thread_id, frame_index)
                raise CrowdFlowError(
                    f"Unexpected error on frame {frame_index}: {exc}"
                ) from exc

        # Augment metadata with runtime provenance (contract §7 / Phase 9).
        if self._runtime is not None:
            metadata["backend"] = self._runtime.backend_name
            metadata["model_format"] = self._runtime.model_format
            metadata["model_version"] = self._runtime.model_version

        result = PipelineResult(
            annotated_frames=annotated_frames,
            timeline=self._timeline.get_timeline(),
            metadata=metadata,
        )
        logger.info("[Thread %s] Exiting CrowdFlowPipeline.run() successfully", thread_id)
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

        # Stage 4c — Graph construction + buffering + inference
        predictions: List[RiskPrediction] = []

        if self._runtime is not None and self._seq_buffer is not None:
            predictions = self._run_inference(
                frame_index, tracks, graph_builder, width, height
            )

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

    def _run_inference(
        self,
        frame_index: int,
        tracks: List[TrackItem],
        graph_builder: GraphBuilder,
        width: int,
        height: int,
    ) -> List[RiskPrediction]:
        """Build a per-frame graph, push it into the buffer, and run inference.

        Args:
            frame_index: Zero-based frame index for logging.
            tracks: Active tracks for this frame.
            graph_builder: Pre-initialised GraphBuilder.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            List of ``RiskPrediction`` objects. Empty during warm-up or on
            inference failure.

        Raises:
            ModelInferenceError: If the runtime raises a non-recoverable error.
        """
        assert self._runtime is not None  # guarded by caller
        assert self._seq_buffer is not None

        # Build per-frame graph (empty graph when no tracks)
        try:
            if tracks:
                positions, velocities = self._extract_arrays(tracks, width, height)
                graph = graph_builder.build(positions, velocities)
            else:
                if not _PYG_AVAILABLE:
                    raise ImportError("torch_geometric not available")
                # Insert a zero-node placeholder graph (contract §6.4)
                graph = Data(
                    x=torch.zeros((0, _NODE_FEATURE_DIM), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, _EDGE_FEATURE_DIM), dtype=torch.float32),
                    num_nodes=0,
                )
        except ImportError:
            logger.warning(
                "torch_geometric not available; skipping graph/inference on frame %d.",
                frame_index,
            )
            return []
        except Exception as exc:
            raise CrowdFlowError(
                f"Graph construction failed on frame {frame_index}: {exc}"
            ) from exc

        # Push graph into sliding-window buffer
        self._seq_buffer.push(graph)

        # During warm-up, not enough frames have accumulated yet
        if not self._seq_buffer.is_ready:
            logger.debug(
                "Frame %d: buffer warming up (%d/%d frames).",
                frame_index,
                self._seq_buffer.current_size,
                self._seq_buffer.window_size,
            )
            return []

        # Assemble flat-tensor batch from the window
        tensor_batch = self._seq_buffer.assemble()

        # Invoke InferenceRuntime
        try:
            result = self._runtime.predict(
                tensor_batch.x,
                tensor_batch.edge_index,
                tensor_batch.edge_attr,
                tensor_batch.batch,
                tensor_batch.seq_lengths,
            )
        except InferenceExecutionError as exc:
            raise ModelInferenceError(
                f"InferenceRuntime failed on frame {frame_index}: {exc}"
            ) from exc

        # Translate InferenceResult → RiskPrediction (contract §7)
        return self._result_to_predictions(result, tracks)

    @staticmethod
    def _result_to_predictions(
        result: InferenceResult,
        tracks: List[TrackItem],
    ) -> List[RiskPrediction]:
        """Translate an ``InferenceResult`` into scene-level ``RiskPrediction`` objects.

        The current model produces a single scene-level classification. Each
        active track receives the same risk label (``region_id = track_id``).

        Args:
            result: An ``InferenceResult`` returned by ``InferenceRuntime``.
            tracks: Active tracks for this frame.

        Returns:
            One ``RiskPrediction`` per active track. Empty list if no tracks.
        """
        if not tracks:
            return []

        if not (0 <= result.predicted_class < len(_CLASS_NAMES)):
            raise ModelInferenceError(
                f"Model returned invalid predicted_class index: {result.predicted_class}"
            )

        label = _CLASS_NAMES[result.predicted_class]
        confidence = result.confidence

        return [
            RiskPrediction(
                region_id=t.track_id,
                label=label,
                confidence=confidence,
            )
            for t in tracks
        ]

    @staticmethod
    def _extract_arrays(
        tracks: List[TrackItem],
        width: int,
        height: int,
    ) -> tuple:
        """Convert TrackItem centroids and velocities to GraphBuilder arrays.

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
