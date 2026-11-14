"""In-memory thread-safe JobManager for asynchronous inference execution.

Manages background thread execution via ThreadPoolExecutor, status tracking,
cancellation tokens, and temporary file cleanup.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from crowdflow_dna.api.schemas.inference import (
    InferenceResultDTO,
    JobResponse,
    JobStatusEnum,
)
from crowdflow_dna.errors import CrowdFlowError

logger = logging.getLogger(__name__)


class JobError(CrowdFlowError):
    """Base exception for job management errors."""


class JobNotFoundError(JobError):
    """Raised when a specified job ID is not found."""


class InvalidJobStateError(JobError):
    """Raised when an operation is invalid for the current job state."""


@dataclass
class JobRecord:
    """Internal state representation for an asynchronous job."""

    job_id: str
    status: JobStatusEnum
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[InferenceResultDTO] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Optional[concurrent.futures.Future[Any]] = None
    temp_files: List[Path] = field(default_factory=list)

    def to_response(self) -> JobResponse:
        """Convert this internal record to a client-facing JobResponse schema."""
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error=self.error,
            result=self.result,
        )


class JobManager:
    """Thread-safe in-memory manager for background inference jobs.

    Coordinates task submission, background execution via a fixed-size
    ThreadPoolExecutor, status queries, graceful cancellation, and automatic
    temporary file cleanup.
    """

    def __init__(
        self,
        max_workers: int = 2,
        inference_service: Optional[Any] = None,
    ) -> None:
        """Initialise the JobManager.

        Args:
            max_workers: Maximum number of concurrent background execution threads.
            inference_service: Optional InferenceService instance. If not provided,
                lazily loaded upon request.
        """
        self._max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="crowddna-inference-worker",
        )
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._inference_service = inference_service

    def submit_job(
        self,
        video_path: Union[str, Path],
        enable_calibration: bool = False,
        task_fn: Optional[Callable[..., Any]] = None,
        temp_files: Optional[List[Union[str, Path]]] = None,
    ) -> JobResponse:
        """Create and enqueue a new inference job.

        Args:
            video_path: Path to the input video file.
            enable_calibration: Flag indicating whether ground-plane calibration is enabled.
            task_fn: Optional custom callable to execute in background thread. If None,
                defaults to InferenceService.run_inference.
            temp_files: Optional list of temporary file paths to track and clean up
                when the job finishes or is cancelled.

        Returns:
            JobResponse with initial status 'queued' and unique job_id.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        video_path_obj = Path(video_path)

        tracked_files: List[Path] = []
        if temp_files:
            tracked_files = [Path(f) for f in temp_files]
        elif video_path_obj.exists():
            tracked_files = [video_path_obj]

        with self._lock:
            record = JobRecord(
                job_id=job_id,
                status=JobStatusEnum.QUEUED,
                created_at=now,
                temp_files=tracked_files,
            )
            self._jobs[job_id] = record

            # Target function to run in background worker
            target = task_fn or self._get_default_task_fn()

            future = self._executor.submit(
                self._worker_wrapper,
                job_id=job_id,
                target=target,
                video_path=str(video_path),
                enable_calibration=enable_calibration,
                cancel_event=record.cancel_event,
            )
            record.future = future
            return record.to_response()

    def _get_default_task_fn(self) -> Callable[..., Any]:
        """Resolve default inference service worker callable."""
        if self._inference_service is not None:
            return self._inference_service.run_inference
        from crowdflow_dna.api.services.inference_service import get_inference_service

        return get_inference_service().run_inference

    def _worker_wrapper(
        self,
        job_id: str,
        target: Callable[..., Any],
        video_path: str,
        enable_calibration: bool,
        cancel_event: threading.Event,
    ) -> None:
        """Background worker execution wrapper with lifecycle and error handling."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.status == JobStatusEnum.CANCELLED or cancel_event.is_set():
                logger.info("Job %s was cancelled before worker started.", job_id)
                self._cleanup_temp_files_locked(record)
                return
            record.status = JobStatusEnum.RUNNING
            record.started_at = datetime.now(timezone.utc)

        try:
            result = target(
                job_id=job_id,
                video_path=video_path,
                enable_calibration=enable_calibration,
                cancel_event=cancel_event,
            )

            with self._lock:
                record = self._jobs.get(job_id)
                if record is not None and record.status != JobStatusEnum.CANCELLED:
                    if cancel_event.is_set():
                        record.status = JobStatusEnum.CANCELLED
                        record.completed_at = datetime.now(timezone.utc)
                    else:
                        record.status = JobStatusEnum.COMPLETED
                        record.completed_at = datetime.now(timezone.utc)
                        if isinstance(result, InferenceResultDTO):
                            record.result = result

        except Exception as exc:
            with self._lock:
                record = self._jobs.get(job_id)
                if record is not None and record.status != JobStatusEnum.CANCELLED:
                    if cancel_event.is_set():
                        record.status = JobStatusEnum.CANCELLED
                        record.completed_at = datetime.now(timezone.utc)
                    else:
                        record.status = JobStatusEnum.FAILED
                        record.completed_at = datetime.now(timezone.utc)
                        record.error = str(exc)
            logger.exception("Job %s failed with exception: %s", job_id, exc)

        finally:
            self.cleanup_temp_files(job_id)

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Retrieve details of a job by ID.

        Args:
            job_id: Unique UUID string identifying the job.

        Returns:
            JobResponse if found, None otherwise.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return record.to_response()

    def cancel_job(self, job_id: str) -> Optional[JobResponse]:
        """Cancel a queued or running job.

        Args:
            job_id: Unique UUID string identifying the job.

        Returns:
            JobResponse with status CANCELLED, or None if the job is not found.

        Raises:
            InvalidJobStateError: If the job has already reached COMPLETED or FAILED state.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            if record.status in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED):
                raise InvalidJobStateError(
                    f"Cannot cancel job '{job_id}' with terminal status '{record.status.value}'."
                )

            if record.status == JobStatusEnum.CANCELLED:
                return record.to_response()

            was_running = record.status == JobStatusEnum.RUNNING

            # Attempt cancelling future if it hasn't started yet
            if record.future is not None and not record.future.running():
                record.future.cancel()

            record.cancel_event.set()
            record.status = JobStatusEnum.CANCELLED
            record.completed_at = datetime.now(timezone.utc)
            if not was_running:
                self._cleanup_temp_files_locked(record)
            return record.to_response()

    def mark_running(self, job_id: str) -> None:
        """Mark a job as RUNNING and record the start timestamp."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None and record.status == JobStatusEnum.QUEUED:
                record.status = JobStatusEnum.RUNNING
                record.started_at = datetime.now(timezone.utc)

    def mark_completed(
        self, job_id: str, result: Optional[InferenceResultDTO] = None
    ) -> None:
        """Mark a job as COMPLETED and attach result data."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None and record.status != JobStatusEnum.CANCELLED:
                record.status = JobStatusEnum.COMPLETED
                record.completed_at = datetime.now(timezone.utc)
                if result is not None:
                    record.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as FAILED and record error message."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None and record.status != JobStatusEnum.CANCELLED:
                record.status = JobStatusEnum.FAILED
                record.completed_at = datetime.now(timezone.utc)
                record.error = error

    def mark_cancelled(self, job_id: str) -> None:
        """Mark a job as CANCELLED."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.cancel_event.set()
                record.status = JobStatusEnum.CANCELLED
                record.completed_at = datetime.now(timezone.utc)

    def is_cancelled(self, job_id: str) -> bool:
        """Check whether a job has been requested to cancel."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            return (
                record.cancel_event.is_set() or record.status == JobStatusEnum.CANCELLED
            )

    def register_temp_file(self, job_id: str, file_path: Union[str, Path]) -> None:
        """Register a temporary file to be cleaned up when the job finishes."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record.temp_files.append(Path(file_path))

    def cleanup_temp_files(self, job_id: str) -> None:
        """Delete all temporary files associated with the job."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            self._cleanup_temp_files_locked(record)

    def _cleanup_temp_files_locked(self, record: JobRecord) -> None:
        """Unlink files safely under lock."""
        remaining: List[Path] = []
        for path in record.temp_files:
            try:
                if path.exists():
                    path.unlink(missing_ok=True)
                    logger.debug("Cleaned up temporary file: %s", path)
            except OSError as exc:
                logger.warning("Failed to delete temp file %s: %s", path, exc)
                remaining.append(path)
        record.temp_files = remaining

    def list_jobs(self) -> List[JobResponse]:
        """List all current jobs in memory."""
        with self._lock:
            return [record.to_response() for record in self._jobs.values()]

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background thread pool executor."""
        with self._lock:
            for record in self._jobs.values():
                if record.status in (JobStatusEnum.QUEUED, JobStatusEnum.RUNNING):
                    record.cancel_event.set()
                    if record.future is not None and not record.future.running():
                        record.future.cancel()
                    was_running = record.status == JobStatusEnum.RUNNING
                    record.status = JobStatusEnum.CANCELLED
                    record.completed_at = datetime.now(timezone.utc)
                    if not was_running:
                        self._cleanup_temp_files_locked(record)
        self._executor.shutdown(wait=wait)


_job_manager: Optional[JobManager] = None
_job_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return the singleton instance of JobManager."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None:
            _job_manager = JobManager()
        return _job_manager


def set_job_manager(manager: Optional[JobManager]) -> None:
    """Set the singleton JobManager instance (used for testing and dependency injection)."""
    global _job_manager
    with _job_manager_lock:
        _job_manager = manager


def reset_job_manager() -> None:
    """Reset the JobManager singleton instance and shut down existing pool."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is not None:
            _job_manager.shutdown(wait=False)
            _job_manager = None
