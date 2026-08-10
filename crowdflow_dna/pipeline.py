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
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np





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

try:
    import torch
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

def get_rss_mb() -> Optional[float]:
    """Return process RSS in MB on Linux, or None if unavailable."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return None

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

    output_video_path: str = ""
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
        
        # Heavy components are deferred until run()
        self._detector: Optional[Yolov8Detector] = None
        self._tracker: Optional[ByteTracker] = None
        self._annotator: Optional[FrameAnnotator] = None
        self._timeline: Optional[TimelineBuilder] = None
        self._runtime: Optional[InferenceRuntime] = None
        self._seq_buffer: Optional[SequenceBuffer] = None

        self._model_path = model_path
        self._model_version = model_version
        self._window_size = window_size

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
            CrowdFlowError: For any other unexpected failure during processing.
        """
        logger.info("Entering CrowdFlowPipeline.run() for video: %s", video_path)

        ingestion_timeout = float(getattr(config, "INGESTION_TIMEOUT_SECONDS", 120))
        deadline = time.monotonic() + ingestion_timeout

        # ----------------------------------------------------------------
        # Stage 1: Ingestion (Streaming Metadata & Iterator)
        # ----------------------------------------------------------------
        frames_iterator, metadata = self._ingestor.load(video_path, deadline)
        logger.info(
            "Ingestion metadata ready: %.1f fps, %dx%d",
            metadata["fps"],
            metadata["width"],
            metadata["height"],
        )

        thread_id = __import__('threading').get_ident()

        # ----------------------------------------------------------------
        # Stage 1.5: Deferred Initialization of Heavy Models
        # ----------------------------------------------------------------
        rss_before_init = get_rss_mb()
        if rss_before_init is not None:
            logger.info("[PIPELINE_MEM] RSS before heavy model init: %.2f MB", rss_before_init)

        if self._detector is None:
            logger.info("Initialising heavy pipeline components (Detector, Tracker, Models)...")
            self._detector = Yolov8Detector()
        if self._tracker is None:
            self._tracker = ByteTracker()
        if self._annotator is None:
            self._annotator = FrameAnnotator()
        if self._timeline is None:
            self._timeline = TimelineBuilder()
            
        if self._model_path is not None and self._runtime is None:
            self._runtime = InferenceRuntime()
            self._runtime.load_model(self._model_path, version=self._model_version)
            self._seq_buffer = SequenceBuffer(window_size=self._window_size)
            logger.info(
                "CrowdFlowPipeline initialised in inference mode: %s (window=%d)",
                self._model_path,
                self._window_size,
            )
        elif self._model_path is None and self._runtime is None:
            logger.info("CrowdFlowPipeline initialised in dummy mode (no model).")

        rss_after_init = get_rss_mb()
        if rss_after_init is not None:
            logger.info("[PIPELINE_MEM] RSS after heavy model init: %.2f MB", rss_after_init)
            if rss_before_init is not None:
                logger.info("[PIPELINE_MEM] Model init memory cost: %.2f MB", rss_after_init - rss_before_init)

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

        # ----------------------------------------------------------------
        # Stage 3.5: Set up FFmpeg Output Encoder
        # ----------------------------------------------------------------
        if not shutil.which("ffmpeg"):
            raise CrowdFlowError("ffmpeg executable not found in PATH for encoding output.")

        # Calculate effective output FPS based on sample rate
        input_fps = float(metadata["fps"])
        sample_rate = int(metadata["sample_rate"])
        output_fps = input_fps / sample_rate if sample_rate > 0 else input_fps
        output_fps = max(1.0, output_fps)

        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        encoder_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(output_fps),
            "-i", "-", # Stdin
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        encoder_process = subprocess.Popen(
            encoder_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        frames_processed = 0
        rss_before_first_frame = get_rss_mb()
        if rss_before_first_frame is not None:
            logger.info("[PIPELINE_MEM] RSS before first frame processing: %.2f MB", rss_before_first_frame)

        # ----------------------------------------------------------------
        # Stage 4: Per-frame Incremental Loop
        # ----------------------------------------------------------------
        try:
            for frame_index, frame in enumerate(frames_iterator):
                if time.monotonic() > deadline:
                    raise CrowdFlowError("Global deadline exceeded during pipeline processing.")

                annotated, predictions = self._process_frame(
                    frame_index, frame, graph_builder, width, height
                )
                
                # Write directly to encoder stdin
                if encoder_process.stdin:
                    try:
                        encoder_process.stdin.write(annotated.tobytes())
                    except BrokenPipeError:
                        stderr_tail = encoder_process.stderr.read().decode(errors="replace") if encoder_process.stderr else ""
                        raise CrowdFlowError(f"Encoder subprocess died unexpectedly. stderr: {stderr_tail}")

                self._timeline.record(frame_index, predictions)
                frames_processed += 1

                # Clean up memory explicitly before next iteration
                del frame
                del annotated
                
                if frames_processed == 1:
                    rss_after_first_frame = get_rss_mb()
                    if rss_after_first_frame is not None:
                        logger.info("[PIPELINE_MEM] RSS after first frame processing: %.2f MB", rss_after_first_frame)

        except CrowdFlowError as exc:
            logger.error("[Thread %s] run() exception handler: error on frame %d: %s", thread_id, frames_processed, exc)
            if encoder_process.poll() is None:
                encoder_process.terminate()
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

        except Exception as exc:
            logger.error("[Thread %s] run() exception handler: unexpected error on frame %d: %s", thread_id, frames_processed, exc)
            if encoder_process.poll() is None:
                encoder_process.terminate()
            # Clean up the partial output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise CrowdFlowError(f"Unexpected error on frame {frames_processed}: {exc}") from exc
            
        finally:
            # ----------------------------------------------------------------
            # Stage 5: Teardown Encoder Process
            # ----------------------------------------------------------------
            if encoder_process.stdin:
                encoder_process.stdin.close()
                
            try:
                # Calculate remaining time for encoder to finish writing MP4 trailing metadata
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    remaining = 1.0 # Give it a small grace period to die
                
                encoder_process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg encoder timed out. Killing process.")
                encoder_process.kill()
                encoder_process.wait()
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise CrowdFlowError("Encoder timed out finalizing the video.")
                
            if encoder_process.stderr:
                encoder_process.stderr.close()

            if isinstance(encoder_process.returncode, int) and encoder_process.returncode != 0:
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise CrowdFlowError(f"FFmpeg encoder failed with exit code {encoder_process.returncode}")

        if frames_processed == 0:
            logger.warning("[Thread %s] No frames processed from iterator.", thread_id)
            if os.path.exists(output_path):
                os.remove(output_path)
            return PipelineResult(metadata=metadata, output_video_path="")

        # Augment metadata with runtime provenance (contract §7 / Phase 9).
        if self._runtime is not None:
            metadata["backend"] = self._runtime.backend_name
            metadata["model_format"] = self._runtime.model_format
            metadata["model_version"] = self._runtime.model_version

        result = PipelineResult(
            output_video_path=output_path,
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
