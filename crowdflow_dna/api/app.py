"""CrowdDNA FastAPI core application factory and configuration.

This module provides the application factory create_app() which configures
the FastAPI application, registers API routes, sets up OpenAPI metadata,
and establishes universal error handling returning standardized ErrorResponse models.
"""

from http import HTTPStatus
import logging
import os
from typing import Any, List, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from crowdflow_dna.api.routes.inference import router as inference_router
from crowdflow_dna.api.routes.system import router as system_router
from crowdflow_dna.api.schemas.common import ErrorDetail, ErrorResponse
from crowdflow_dna.errors import (
    CrowdFlowError,
    InvalidVideoFormatError,
    ModelInferenceError,
    UploadSizeExceededError,
    VideoCorruptionError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register universal exception handlers on the FastAPI application.

    Ensures that all client and server errors return standardized
    ErrorResponse JSON payloads with appropriate HTTP status codes.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle standard HTTP exceptions with unified ErrorResponse format."""
        try:
            phrase = HTTPStatus(exc.status_code).phrase
        except ValueError:
            phrase = "HTTP Error"

        detail_val: Union[str, List[ErrorDetail]]
        if isinstance(exc.detail, list):
            detail_val = [
                ErrorDetail(
                    loc=d.get("loc"),
                    msg=str(d.get("msg", d)),
                    type=d.get("type"),
                )
                if isinstance(d, dict)
                else ErrorDetail(msg=str(d))
                for d in exc.detail
            ]
        else:
            detail_val = str(exc.detail) if exc.detail is not None else phrase

        payload = ErrorResponse(
            detail=detail_val,
            error=phrase,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI request parameter and body validation failures."""
        details: List[ErrorDetail] = []
        for err in exc.errors():
            details.append(
                ErrorDetail(
                    loc=list(err.get("loc", [])),
                    msg=str(err.get("msg", "Validation error")),
                    type=str(err.get("type", "value_error")),
                )
            )

        payload = ErrorResponse(
            detail=details,
            error="Validation Error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic internal model validation failures."""
        details: List[ErrorDetail] = [
            ErrorDetail(
                loc=list(err.get("loc", [])),
                msg=str(err.get("msg", "Validation error")),
                type=str(err.get("type", "value_error")),
            )
            for err in exc.errors()
        ]
        payload = ErrorResponse(
            detail=details,
            error="Validation Error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(CrowdFlowError)
    async def crowdflow_domain_exception_handler(
        request: Request, exc: CrowdFlowError
    ) -> JSONResponse:
        """Map CrowdFlow domain exceptions to appropriate HTTP error responses."""
        if isinstance(exc, UploadSizeExceededError):
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif isinstance(exc, (InvalidVideoFormatError, VideoCorruptionError)):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, ModelInferenceError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            status_code = status.HTTP_400_BAD_REQUEST

        try:
            phrase = HTTPStatus(status_code).phrase
        except ValueError:
            phrase = "CrowdFlow Error"

        payload = ErrorResponse(
            detail=str(exc) or phrase,
            error=phrase,
            status_code=status_code,
        )
        return JSONResponse(
            status_code=status_code,
            content=payload.model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled internal server exceptions."""
        logger.exception("Unhandled server exception encountered: %s", exc)
        payload = ErrorResponse(
            detail="Internal server error",
            error="Internal Server Error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )


def create_app(
    title: str = "CrowdDNA API",
    version: str = "1.0.0",
    description: str = (
        "CrowdDNA Real-Time Pedestrian Dynamics and Risk Prediction REST API. "
        "Provides high-performance crowd risk estimation, telemetry ingestion, "
        "and operational health monitoring."
    ),
    **kwargs: Any,
) -> FastAPI:
    """Create and configure a new CrowdDNA FastAPI application instance.

    Initializes application metadata, registers system and operational routers,
    initializes lightweight application state without eager ML model loading,
    and binds global exception handlers returning standardized ErrorResponse models.

    Args:
        title: API title for documentation and OpenAPI schema.
        version: API semantic version string.
        description: API description summary.
        **kwargs: Additional keyword arguments forwarded to the FastAPI constructor.

    Returns:
        Configured FastAPI application instance.
    """
    openapi_tags = [
        {
            "name": "System",
            "description": "System health, readiness, liveness probes, and engine version metadata.",
        },
        {
            "name": "Inference",
            "description": "Asynchronous video inference job submission, polling, and lifecycle management.",
        },
    ]

    app = FastAPI(
        title=title,
        version=version,
        description=description,
        openapi_tags=openapi_tags,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        **kwargs,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    # Allows environment-based configuration via CROWDDNA_CORS_ORIGINS with sensible localhost defaults
    cors_env = os.environ.get("CROWDDNA_CORS_ORIGINS")
    if cors_env is not None and cors_env.strip():
        raw_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        allow_origins = ["*"] if "*" in raw_origins else raw_origins
    else:
        allow_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lightweight application state initialization (no eager model loading)
    app.state.model_loaded = False
    app.state.device = None

    # Register routers
    app.include_router(system_router)
    app.include_router(inference_router)

    # Register universal exception handlers
    register_exception_handlers(app)

    return app


# Module-level default application instance for ASGI servers
app = create_app()
