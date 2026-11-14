"""Data models and response types for the CrowdDNA SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class RiskPrediction:
    """Represents a crowd risk prediction for a detected pedestrian/region.

    Attributes:
        region_id: Identifier of the tracked pedestrian or region.
        label: Qualitative classification label ("Safe", "Congesting", "Critical").
        confidence: Confidence score of the prediction in range [0.0, 1.0].
    """

    region_id: int
    label: str
    confidence: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RiskPrediction:
        """Instantiate RiskPrediction from a dictionary.

        Args:
            data: Dictionary containing risk prediction attributes.

        Returns:
            RiskPrediction instance.
        """
        return cls(
            region_id=int(data.get("region_id", 0)),
            label=str(data.get("label", "Safe")),
            confidence=float(data.get("confidence", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction to dictionary.

        Returns:
            Serialized dictionary.
        """
        return asdict(self)


@dataclass
class TimelineEntry:
    """Represents risk prediction state for a single video frame.

    Attributes:
        frame_index: Zero-based frame index.
        predictions: List of risk predictions for pedestrians in this frame.
    """

    frame_index: int
    predictions: list[RiskPrediction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimelineEntry:
        """Instantiate TimelineEntry from a dictionary.

        Args:
            data: Dictionary containing timeline entry attributes.

        Returns:
            TimelineEntry instance.
        """
        raw_predictions = data.get("predictions", [])
        predictions = [
            p if isinstance(p, RiskPrediction) else RiskPrediction.from_dict(p)
            for p in raw_predictions
        ]
        return cls(
            frame_index=int(data.get("frame_index", 0)),
            predictions=predictions,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert timeline entry to dictionary.

        Returns:
            Serialized dictionary.
        """
        return {
            "frame_index": self.frame_index,
            "predictions": [p.to_dict() for p in self.predictions],
        }


@dataclass
class VideoMetadata:
    """Metadata describing the processed video stream.

    Attributes:
        fps: Frames per second of the video.
        width: Horizontal resolution in pixels.
        height: Vertical resolution in pixels.
        frame_count: Total frame count.
        duration_seconds: Total duration in seconds.
        sample_rate: Pipeline frame sampling rate.
        extra: Additional arbitrary video metadata fields.
    """

    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoMetadata:
        """Instantiate VideoMetadata from a dictionary.

        Args:
            data: Dictionary containing video metadata.

        Returns:
            VideoMetadata instance.
        """
        known_keys = {
            "fps",
            "width",
            "height",
            "frame_count",
            "duration_seconds",
            "sample_rate",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            fps=float(data.get("fps", 0.0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            frame_count=int(data.get("frame_count", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            sample_rate=int(data.get("sample_rate", 1)),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert video metadata to dictionary.

        Returns:
            Serialized dictionary.
        """
        result: dict[str, Any] = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
        }
        result.update(self.extra)
        return result


@dataclass
class InferenceResult:
    """Inference execution result produced by the pipeline.

    Attributes:
        timeline: Frame-by-frame risk timeline.
        metadata: Video metadata for processed footage.
        output_video_url: Downloadable URL or path for the annotated video.
        predicted_class: Overall scene risk classification index.
        probabilities: Classification probability distribution.
        confidence: Overall prediction confidence.
        backend: Runtime backend used for inference (e.g. PyTorch, ONNX).
        inference_time_ms: Total inference latency in milliseconds.
        model_format: Format of the deployed model (e.g. "pt", "onnx").
        model_version: Version identifier of the model.
        raw_data: Unmodified raw response data from the API.
    """

    timeline: list[TimelineEntry] = field(default_factory=list)
    metadata: VideoMetadata | None = None
    output_video_url: str | None = None
    predicted_class: int = 0
    probabilities: list[float] = field(default_factory=list)
    confidence: float = 0.0
    backend: str = ""
    inference_time_ms: float = 0.0
    model_format: str = ""
    model_version: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InferenceResult:
        """Instantiate InferenceResult from a dictionary.

        Args:
            data: Response payload from the inference endpoint.

        Returns:
            InferenceResult instance.
        """
        timeline_data = data.get("timeline", [])
        timeline = [
            t if isinstance(t, TimelineEntry) else TimelineEntry.from_dict(t)
            for t in timeline_data
        ]
        meta_data = data.get("metadata")
        metadata = (
            meta_data
            if isinstance(meta_data, VideoMetadata)
            else VideoMetadata.from_dict(meta_data)
            if isinstance(meta_data, Mapping)
            else None
        )
        return cls(
            timeline=timeline,
            metadata=metadata,
            output_video_url=data.get("output_video_url")
            or data.get("output_video_path"),
            predicted_class=int(data.get("predicted_class", 0)),
            probabilities=[float(p) for p in data.get("probabilities", [])],
            confidence=float(data.get("confidence", 0.0)),
            backend=str(data.get("backend", "")),
            inference_time_ms=float(data.get("inference_time_ms", 0.0)),
            model_format=str(data.get("model_format", "")),
            model_version=data.get("model_version"),
            raw_data=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert inference result to dictionary.

        Returns:
            Serialized dictionary.
        """
        return {
            "timeline": [t.to_dict() for t in self.timeline],
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "output_video_url": self.output_video_url,
            "predicted_class": self.predicted_class,
            "probabilities": list(self.probabilities),
            "confidence": self.confidence,
            "backend": self.backend,
            "inference_time_ms": self.inference_time_ms,
            "model_format": self.model_format,
            "model_version": self.model_version,
            "raw_data": dict(self.raw_data),
        }


# Alias JobResult to InferenceResult for naming compatibility
JobResult = InferenceResult


@dataclass
class JobStatus:
    """Status and result of an asynchronous processing job.

    Attributes:
        job_id: Unique identifier of the job.
        status: Current status ("PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED").
        progress: Completion progress percentage between 0.0 and 1.0.
        result: Completed inference result if available.
        error: Error message if the job failed.
        created_at: ISO timestamp of job submission.
        completed_at: ISO timestamp of job completion.
    """

    job_id: str
    status: str
    progress: float = 0.0
    result: InferenceResult | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None

    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.status.upper() in ("COMPLETED", "SUCCESS", "SUCCEEDED")

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status.upper() in ("FAILED", "ERROR")

    @property
    def is_cancelled(self) -> bool:
        """Check if job was cancelled."""
        return self.status.upper() in ("CANCELLED", "CANCELED")

    @property
    def is_running(self) -> bool:
        """Check if job is actively pending or processing."""
        return self.status.upper() in ("PENDING", "PROCESSING", "RUNNING", "QUEUED")

    @property
    def is_done(self) -> bool:
        """Check if job has reached any terminal state."""
        return self.is_completed or self.is_failed or self.is_cancelled

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobStatus:
        """Instantiate JobStatus from a dictionary.

        Args:
            data: Dictionary returned by the jobs endpoint.

        Returns:
            JobStatus instance.
        """
        res_data = data.get("result")
        result = (
            res_data
            if isinstance(res_data, InferenceResult)
            else InferenceResult.from_dict(res_data)
            if isinstance(res_data, Mapping)
            else None
        )
        return cls(
            job_id=str(data.get("job_id", "")),
            status=str(data.get("status", "UNKNOWN")),
            progress=float(data.get("progress", 0.0)),
            result=result,
            error=data.get("error"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert job status to dictionary.

        Returns:
            Serialized dictionary.
        """
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class HealthStatus:
    """Overall system health status.

    Attributes:
        status: High-level health state (e.g. "ok", "degraded", "unhealthy").
        startup: Startup probe status or details.
        readiness: Readiness probe status or details.
        liveness: Liveness probe status or details.
        details: Additional probe telemetry and metadata.
    """

    status: str = "ok"
    startup: bool | dict[str, Any] = True
    readiness: bool | dict[str, Any] = True
    liveness: bool | dict[str, Any] = True
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str | None:
        """Convenience property for API version string if present."""
        return self.details.get("version")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HealthStatus:
        """Instantiate HealthStatus from dictionary.

        Args:
            data: Health endpoint response payload.

        Returns:
            HealthStatus instance.
        """
        return cls(
            status=str(data.get("status", "ok")),
            startup=data.get("startup", True),
            readiness=data.get("readiness", True),
            liveness=data.get("liveness", True),
            details={
                k: v
                for k, v in data.items()
                if k not in {"status", "startup", "readiness", "liveness"}
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert health status to dictionary.

        Returns:
            Serialized dictionary.
        """
        res: dict[str, Any] = {
            "status": self.status,
            "startup": self.startup,
            "readiness": self.readiness,
            "liveness": self.liveness,
        }
        res.update(self.details)
        return res


@dataclass
class LiveStatus(Mapping[str, Any]):
    """Liveness probe status model supporting dict-like access.

    Attributes:
        alive: Boolean indicating whether the process is alive.
        status: Status string (default "ok").
        details: Additional diagnostic attributes.
    """

    alive: bool = True
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "alive":
            return self.alive
        if key == "status":
            return self.status
        return self.details[key]

    def __iter__(self) -> Iterator[str]:
        yield "alive"
        yield "status"
        yield from self.details

    def __len__(self) -> int:
        return 2 + len(self.details)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiveStatus:
        """Instantiate LiveStatus from dictionary.

        Args:
            data: Liveness endpoint response payload.

        Returns:
            LiveStatus instance.
        """
        status = str(data.get("status", "ok"))
        if "alive" in data:
            alive = bool(data["alive"])
        else:
            alive = status.lower() in ("ok", "alive", "healthy", "true")
        details = {k: v for k, v in data.items() if k not in {"alive", "status"}}
        return cls(alive=alive, status=status, details=details)

    def to_dict(self) -> dict[str, Any]:
        """Convert liveness status to dictionary.

        Returns:
            Serialized dictionary.
        """
        res: dict[str, Any] = {"alive": self.alive, "status": self.status}
        res.update(self.details)
        return res


@dataclass
class ReadyStatus(Mapping[str, Any]):
    """Readiness probe status model supporting dict-like access.

    Attributes:
        ready: Boolean indicating whether the service is ready for traffic.
        status: Status string (default "ok").
        details: Additional diagnostic attributes.
    """

    ready: bool = True
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "ready":
            return self.ready
        if key == "status":
            return self.status
        return self.details[key]

    def __iter__(self) -> Iterator[str]:
        yield "ready"
        yield "status"
        yield from self.details

    def __len__(self) -> int:
        return 2 + len(self.details)

    @property
    def model_loaded(self) -> bool:
        """Check whether the model is reported as loaded."""
        return bool(self.details.get("model_loaded", False))

    @property
    def device(self) -> str | None:
        """Device name from readiness details if available."""
        return self.details.get("device")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReadyStatus:
        """Instantiate ReadyStatus from dictionary.

        Args:
            data: Readiness endpoint response payload.

        Returns:
            ReadyStatus instance.
        """
        status = str(data.get("status", "ok"))
        if "ready" in data:
            ready = bool(data["ready"])
        else:
            ready = status.lower() in ("ok", "ready", "healthy", "true")
        details = {k: v for k, v in data.items() if k not in {"ready", "status"}}
        return cls(ready=ready, status=status, details=details)

    def to_dict(self) -> dict[str, Any]:
        """Convert readiness status to dictionary.

        Returns:
            Serialized dictionary.
        """
        res: dict[str, Any] = {"ready": self.ready, "status": self.status}
        res.update(self.details)
        return res


@dataclass
class VersionInfo:
    """API and core engine version metadata.

    Attributes:
        version: Application semantic version.
        api_version: API schema version.
        core_version: Core pipeline engine version if available.
        git_commit: Git commit hash if built in CI.
        build_date: Build date or release timestamp.
        details: Additional version metadata.
    """

    version: str = "1.0.0"
    api_version: str = "v1"
    core_version: str | None = None
    git_commit: str | None = None
    build_date: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def engine_version(self) -> str | None:
        """Alias for core engine version."""
        return self.core_version

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VersionInfo:
        """Instantiate VersionInfo from dictionary.

        Args:
            data: Version endpoint response payload.

        Returns:
            VersionInfo instance.
        """
        known_keys = {
            "version",
            "api_version",
            "core_version",
            "engine_version",
            "git_commit",
            "build_date",
        }
        details = {k: v for k, v in data.items() if k not in known_keys}
        core_version = data.get("core_version") or data.get("engine_version")
        return cls(
            version=str(data.get("version", "1.0.0")),
            api_version=str(data.get("api_version", "v1")),
            core_version=str(core_version) if core_version is not None else None,
            git_commit=data.get("git_commit"),
            build_date=data.get("build_date"),
            details=details,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert version info to dictionary.

        Returns:
            Serialized dictionary.
        """
        res: dict[str, Any] = {
            "version": self.version,
            "api_version": self.api_version,
            "core_version": self.core_version,
            "git_commit": self.git_commit,
            "build_date": self.build_date,
        }
        res.update(self.details)
        return res
