"""System status and diagnostic endpoints for CrowdDNA API.

Provides health checks, Kubernetes-style liveness and readiness probes,
and version metadata. The readiness probe inspects system capability
and device availability without eagerly loading heavy ML models.
"""

from datetime import datetime, timezone
import logging
import os

from fastapi import APIRouter, Request, status

from crowdflow_dna.api.schemas.common import (
    ErrorResponse,
    HealthStatus,
    LiveStatus,
    ReadyStatus,
    VersionInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["System"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Bad request error response.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error response.",
        },
    },
)


def _check_model_loaded(request: Request) -> bool:
    """Check whether ML models are loaded in memory without eager loading.

    Inspects application state attributes to determine if inference models
    have already been instantiated, avoiding expensive imports or disk I/O.

    Args:
        request: The incoming FastAPI request.

    Returns:
        True if an ML model is loaded in memory, False otherwise.
    """
    app_state = getattr(request.app, "state", None)
    if app_state is None:
        return False

    if getattr(app_state, "model_loaded", None) is not None:
        return bool(app_state.model_loaded)

    if getattr(app_state, "model", None) is not None:
        return True

    pipeline = getattr(app_state, "pipeline", None)
    if pipeline is not None:
        if getattr(pipeline, "inference_runtime", None) is not None:
            return True
        if getattr(pipeline, "has_model", False):
            return True

    return False


def _resolve_device(request: Request) -> str:
    """Resolve compute hardware device string safely.

    Checks application state, environment overrides, and the safe device
    probe without triggering full model instantiation.

    Args:
        request: The incoming FastAPI request.

    Returns:
        Device identifier string (e.g., 'cpu' or 'cuda').
    """
    app_state = getattr(request.app, "state", None)
    if app_state is not None and getattr(app_state, "device", None) is not None:
        return str(app_state.device)

    env_device = os.environ.get("CROWDDNA_DEVICE")
    if env_device:
        return env_device.strip()

    try:
        from crowdflow_dna.device import get_safe_device

        return str(get_safe_device())
    except Exception as exc:
        logger.debug("Device resolution fallback to cpu: %s", exc)
        return "cpu"


@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Application health check",
    description="Returns high-level health status, current timestamp, and API version.",
)
async def get_health() -> HealthStatus:
    """Return application health status and UTC timestamp.

    Returns:
        HealthStatus containing status, current UTC timestamp, and version.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
    )


@router.get(
    "/health/live",
    response_model=LiveStatus,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns process liveness indicator for orchestrators.",
)
async def get_health_live() -> LiveStatus:
    """Return process liveness confirmation.

    Returns:
        LiveStatus indicating the process is alive.
    """
    return LiveStatus(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadyStatus,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Checks service readiness, model load status, and device without eager model loading.",
)
async def get_health_ready(request: Request) -> ReadyStatus:
    """Return readiness status, model load flag, and hardware device.

    Args:
        request: The incoming FastAPI request.

    Returns:
        ReadyStatus with readiness, model_loaded flag, and device string.
    """
    model_loaded = _check_model_loaded(request)
    device = _resolve_device(request)
    return ReadyStatus(
        status="ready",
        model_loaded=model_loaded,
        device=device,
    )


@router.get(
    "/version",
    response_model=VersionInfo,
    status_code=status.HTTP_200_OK,
    summary="Version metadata",
    description="Returns the API specification version and CrowdFlow DNA engine version.",
)
async def get_version() -> VersionInfo:
    """Return API and engine version metadata.

    Returns:
        VersionInfo with api_version and engine_version.
    """
    try:
        import crowdflow_dna

        engine_version = getattr(crowdflow_dna, "__version__", "1.0.0")
    except Exception:
        engine_version = "1.0.0"

    return VersionInfo(
        api_version="1.0.0",
        engine_version=engine_version,
    )
