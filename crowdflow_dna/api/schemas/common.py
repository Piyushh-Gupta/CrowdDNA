"""Common Pydantic v2 schemas for CrowdDNA API.

This module defines standardized request and response data models
used across system and operational endpoints, including health checks,
liveness/readiness probes, version information, and error responses.
"""

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(BaseModel):
    """Health check status response model.

    Attributes:
        status: Overall application health indicator string.
        timestamp: UTC timestamp when the health check was performed (ISO 8601).
        version: API version string.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="healthy",
        description="Overall application health indicator string.",
    )
    timestamp: str = Field(
        ...,
        description="UTC timestamp in ISO 8601 format.",
    )
    version: str = Field(
        default="1.0.0",
        description="API version string.",
    )


class LiveStatus(BaseModel):
    """Liveness probe response model.

    Used by container orchestrators to verify the process is responsive
    and running.

    Attributes:
        status: Liveness indicator string.
    """

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="alive",
        description="Liveness status indicator.",
    )


class ReadyStatus(BaseModel):
    """Readiness probe response model.

    Used by orchestrators to determine whether the service is ready
    to accept incoming traffic without eagerly loading heavy ML models.

    Attributes:
        status: Readiness indicator string.
        model_loaded: Flag indicating if ML models are loaded in memory.
        device: Hardware compute device string (e.g., 'cpu' or 'cuda').
    """

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="ready",
        description="Readiness status indicator.",
    )
    model_loaded: bool = Field(
        ...,
        description="Flag indicating if inference models are currently loaded in memory.",
    )
    device: str = Field(
        ...,
        description="Hardware compute device identifier.",
    )


class VersionInfo(BaseModel):
    """API and core engine version response model.

    Attributes:
        api_version: Version of the CrowdDNA HTTP API.
        engine_version: Version of the underlying CrowdFlow DNA core package.
    """

    model_config = ConfigDict(extra="ignore")

    api_version: str = Field(
        default="1.0.0",
        description="Version of the CrowdDNA HTTP API.",
    )
    engine_version: str = Field(
        default="1.0.0",
        description="Version of the underlying CrowdFlow DNA core engine.",
    )


class ErrorDetail(BaseModel):
    """Detailed description of an individual error or validation failure.

    Attributes:
        loc: Location path of the invalid parameter or body field.
        msg: Human-readable error message.
        type: Machine-readable error classification type.
    """

    model_config = ConfigDict(extra="ignore")

    loc: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Location path of the invalid parameter or field.",
    )
    msg: str = Field(
        ...,
        description="Human-readable error description.",
    )
    type: Optional[str] = Field(
        default=None,
        description="Machine-readable error type identifier.",
    )


class ErrorResponse(BaseModel):
    """Universal error response model for HTTP API errors.

    Attributes:
        detail: Human-readable error description or list of detailed error objects.
        error: Machine-readable error category or HTTP status phrase.
        status_code: HTTP status code corresponding to the error.
    """

    model_config = ConfigDict(extra="ignore")

    detail: Union[str, List[ErrorDetail]] = Field(
        ...,
        description="Error detail message or list of detailed validation errors.",
    )
    error: str = Field(
        ...,
        description="Machine-readable error classification or status phrase.",
    )
    status_code: int = Field(
        ...,
        description="HTTP status code.",
    )
