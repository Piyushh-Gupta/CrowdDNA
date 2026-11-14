"""FastAPI route handlers for asynchronous inference execution."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from crowdflow_dna.api.jobs.manager import (
    InvalidJobStateError,
    JobManager,
    get_job_manager,
)
from crowdflow_dna.api.schemas.inference import JobResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])

ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi"})


@router.post(
    "/inference",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit video for asynchronous crowd inference",
    description="Accepts an MP4 or AVI video file and queues an asynchronous inference job.",
)
async def submit_inference(
    file: UploadFile = File(..., description="Uploaded video file (.mp4 or .avi)"),
    enable_calibration: bool = Form(
        False, description="Whether to apply ground-plane metric calibration"
    ),
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Handle video upload and enqueue an asynchronous inference job.

    Args:
        file: Multipart uploaded video file.
        enable_calibration: Optional boolean flag to enable metric calibration.
        job_manager: Injected JobManager instance.

    Returns:
        JobResponse with status 'queued' and unique job_id.

    Raises:
        HTTPException: 400 if the video file format is unsupported.
        HTTPException: 500 if storing the uploaded file fails.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    upload_dir = Path(tempfile.gettempdir()) / "crowddna_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_path_str = tempfile.mkstemp(
        prefix="crowddna_upload_",
        suffix=suffix,
        dir=str(upload_dir),
    )
    temp_path = Path(temp_path_str)

    try:
        with open(fd, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.exception(
            "Failed to write uploaded file to temporary path %s", temp_path
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store uploaded video: {exc}",
        ) from exc

    try:
        return job_manager.submit_job(
            video_path=temp_path,
            enable_calibration=enable_calibration,
            temp_files=[temp_path],
        )
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.exception("Failed to enqueue job with JobManager: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue inference job: {exc}",
        ) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get status and results of an inference job",
)
def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Retrieve details and results for a specific inference job.

    Args:
        job_id: UUID string identifying the job.
        job_manager: Injected JobManager instance.

    Returns:
        JobResponse containing current status, timestamps, and results if completed.

    Raises:
        HTTPException: 404 if the job_id does not exist.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job


@router.delete(
    "/jobs/{job_id}",
    response_model=JobResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel a queued or running inference job",
)
def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    """Cancel a queued or running inference job.

    Args:
        job_id: UUID string identifying the job.
        job_manager: Injected JobManager instance.

    Returns:
        JobResponse with status 'cancelled'.

    Raises:
        HTTPException: 404 if the job_id does not exist.
        HTTPException: 400 if the job is already completed or failed.
    """
    try:
        job = job_manager.cancel_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found.",
            )
        return job
    except InvalidJobStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
