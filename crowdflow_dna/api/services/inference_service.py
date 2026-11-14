"""Inference service bridging JobManager and CrowdFlowPipeline.

Executes crowd analysis vision pipelines in background workers,
converts pipeline domain models to API DTOs, and updates job state.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Optional

from crowdflow_dna.api.schemas.inference import (
    InferenceResultDTO,
    RiskPredictionDTO,
    TimelineEntryDTO,
    VideoMetadataDTO,
)
from crowdflow_dna.calibration import CalibrationConfig
from crowdflow_dna.pipeline import CrowdFlowPipeline

logger = logging.getLogger(__name__)


class InferenceService:
    """Service that executes crowd inference pipelines and reports status to JobManager."""

    def __init__(
        self,
        job_manager: Optional[Any] = None,
        pipeline_factory: Optional[Callable[..., CrowdFlowPipeline]] = None,
    ) -> None:
        """Initialise the inference service.

        Args:
            job_manager: Optional JobManager instance. If None, resolves the singleton.
            pipeline_factory: Callable that creates a CrowdFlowPipeline instance.
        """
        self._job_manager = job_manager
        self._pipeline_factory = pipeline_factory or CrowdFlowPipeline

    @property
    def job_manager(self) -> Any:
        """Return the JobManager instance."""
        if self._job_manager is None:
            from crowdflow_dna.api.jobs.manager import get_job_manager

            self._job_manager = get_job_manager()
        return self._job_manager

    def run_inference(
        self,
        job_id: str,
        video_path: str,
        enable_calibration: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[InferenceResultDTO]:
        """Execute the CrowdFlowPipeline for a given job.

        Args:
            job_id: Unique identifier for the job.
            video_path: Path to the temporary video file to process.
            enable_calibration: Whether to apply ground-plane metric calibration.
            cancel_event: Optional threading event indicating cancellation.

        Returns:
            InferenceResultDTO if successful, or None if cancelled.

        Raises:
            Exception: Any exception raised by CrowdFlowPipeline during processing.
        """
        logger.info("Starting inference execution for job %s on %s", job_id, video_path)

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Job %s cancelled before pipeline execution.", job_id)
            return None

        self.job_manager.mark_running(job_id)

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Job %s cancelled after mark_running.", job_id)
            return None

        # Build calibration config
        if enable_calibration:
            calibration_config = CalibrationConfig(
                enabled=True,
                image_points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                world_points_m=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            )
            calibration_config.validate()
        else:
            calibration_config = CalibrationConfig(enabled=False)

        model_path = os.environ.get("CROWDDNA_MODEL_PATH")

        pipeline = self._pipeline_factory(
            model_path=model_path,
            calibration_config=calibration_config,
        )

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Job %s cancelled right before pipeline.run.", job_id)
            return None

        try:
            pipeline_result = pipeline.run(video_path)
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Job %s cancelled during pipeline execution.", job_id)
                return None
            self.job_manager.mark_failed(job_id, str(exc))
            raise

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Job %s cancelled immediately after pipeline.run.", job_id)
            return None

        result_dto = self.convert_result_to_dto(pipeline_result)
        self.job_manager.mark_completed(job_id, result_dto)
        return result_dto

    @staticmethod
    def convert_result_to_dto(pipeline_result: Any) -> InferenceResultDTO:
        """Convert a pipeline PipelineResult into an InferenceResultDTO.

        Args:
            pipeline_result: PipelineResult or compatible object.

        Returns:
            InferenceResultDTO containing timeline, metadata, and output video path.
        """
        timeline_entries: list[TimelineEntryDTO] = []
        raw_timeline = getattr(pipeline_result, "timeline", []) or []
        for entry in raw_timeline:
            preds: list[RiskPredictionDTO] = []
            for pred in getattr(entry, "predictions", []) or []:
                preds.append(
                    RiskPredictionDTO(
                        region_id=int(getattr(pred, "region_id", 0)),
                        label=str(getattr(pred, "label", "Safe")),
                        confidence=float(getattr(pred, "confidence", 0.0)),
                    )
                )
            timeline_entries.append(
                TimelineEntryDTO(
                    frame_index=int(getattr(entry, "frame_index", 0)),
                    predictions=preds,
                )
            )

        meta_raw = getattr(pipeline_result, "metadata", {}) or {}
        metadata_dto = VideoMetadataDTO(
            fps=float(meta_raw.get("fps", 0.0)),
            width=int(meta_raw.get("width", 0)),
            height=int(meta_raw.get("height", 0)),
            frame_count=int(meta_raw.get("frame_count", 0)),
            duration_seconds=float(meta_raw.get("duration_seconds", 0.0)),
            sample_rate=int(meta_raw.get("sample_rate", 1)),
        )

        output_path = getattr(pipeline_result, "output_video_path", None)
        return InferenceResultDTO(
            output_video_path=str(output_path) if output_path else None,
            timeline=timeline_entries,
            metadata=metadata_dto,
        )


_service_singleton: Optional[InferenceService] = None
_service_lock = threading.Lock()


def get_inference_service() -> InferenceService:
    """Return the global InferenceService singleton instance."""
    global _service_singleton
    with _service_lock:
        if _service_singleton is None:
            _service_singleton = InferenceService()
        return _service_singleton


def set_inference_service(service: Optional[InferenceService]) -> None:
    """Set the global InferenceService instance (for testing/mocking)."""
    global _service_singleton
    with _service_lock:
        _service_singleton = service
