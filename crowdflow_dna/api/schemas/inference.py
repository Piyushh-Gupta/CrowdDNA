"""Pydantic schemas for the asynchronous inference execution API.

Defines request/response models and data transfer objects representing
job lifecycles, risk predictions, timelines, and inference results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobStatusEnum(str, Enum):
    """Lifecycle states for asynchronous inference jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Convenience alias
JobStatus = JobStatusEnum


class RiskPredictionDTO(BaseModel):
    """Data transfer object for regional crowd risk prediction."""

    model_config = ConfigDict(from_attributes=True)

    region_id: int = Field(..., description="Unique integer ID of the spatial region.")
    label: str = Field(
        ..., description="Risk classification label: Safe, Congesting, or Critical."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score between 0.0 and 1.0."
    )


class TimelineEntryDTO(BaseModel):
    """Data transfer object for a single frame's accumulated risk predictions."""

    model_config = ConfigDict(from_attributes=True)

    frame_index: int = Field(..., ge=0, description="Zero-based frame index.")
    predictions: List[RiskPredictionDTO] = Field(
        default_factory=list,
        description="List of risk predictions across regions in this frame.",
    )


class VideoMetadataDTO(BaseModel):
    """Data transfer object for processed video stream metadata."""

    model_config = ConfigDict(from_attributes=True, extra="allow")

    fps: float = Field(default=0.0, description="Frames per second of the video.")
    width: int = Field(default=0, description="Width of the video in pixels.")
    height: int = Field(default=0, description="Height of the video in pixels.")
    frame_count: int = Field(default=0, description="Total frame count in the video.")
    duration_seconds: float = Field(
        default=0.0, description="Total duration in seconds."
    )
    sample_rate: int = Field(
        default=1, description="Sampling rate applied during frame ingestion."
    )


class InferenceResultDTO(BaseModel):
    """Data transfer object containing the completed inference output."""

    model_config = ConfigDict(from_attributes=True, extra="allow")

    output_video_path: Optional[str] = Field(
        default=None,
        description="Filesystem path to the annotated output video file.",
    )
    timeline: List[TimelineEntryDTO] = Field(
        default_factory=list,
        description="Per-frame risk predictions accumulated across the video.",
    )
    metadata: Optional[VideoMetadataDTO] = Field(
        default=None,
        description="Video stream metadata extracted during processing.",
    )


class JobResponse(BaseModel):
    """Schema representing the status and details of an inference job."""

    model_config = ConfigDict(from_attributes=True, extra="allow")

    job_id: str = Field(..., description="Unique UUID identifier for the job.")
    status: JobStatusEnum = Field(
        ..., description="Current lifecycle state of the job."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the job was accepted into the queue.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the background worker began executing.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the job reached terminal state.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description if the job failed.",
    )
    result: Optional[InferenceResultDTO] = Field(
        default=None,
        description="Inference output data when the job has completed.",
    )
