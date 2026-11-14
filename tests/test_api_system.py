"""Unit tests for CrowdDNA API application factory and system endpoints.

Tests all system endpoints (/health, /health/live, /health/ready, /version),
OpenAPI schema generation, docs endpoints, and universal error responses
for HTTP exceptions, validation errors, domain errors, and unhandled server errors.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

# Ensure repository root is on sys.path for test execution
REPO_ROOT = str(Path(__file__).parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crowdflow_dna.api import create_app  # noqa: E402
from crowdflow_dna.api.schemas.common import (  # noqa: E402
    ErrorDetail,
    ErrorResponse,
    HealthStatus,
    LiveStatus,
    ReadyStatus,
    VersionInfo,
)
from crowdflow_dna.errors import (  # noqa: E402
    CrowdFlowError,
    InvalidVideoFormatError,
    ModelInferenceError,
    UploadSizeExceededError,
    VideoCorruptionError,
)


@pytest.fixture
def app():
    """Create a fresh FastAPI test application instance."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a Starlette TestClient with the test application."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Application Factory Tests
# ---------------------------------------------------------------------------


def test_create_app_factory(app):
    """Verify application factory configuration and initial state."""
    assert app.title == "CrowdDNA API"
    assert app.version == "1.0.0"
    assert app.state.model_loaded is False
    assert app.state.device is None


def test_api_import_remains_lightweight():
    """Importing crowdflow_dna.api must not eagerly import heavy ML libraries."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "import crowdflow_dna.api\n"
        "bad_modules = [m for m in sys.modules if m in ('torch', 'torch_geometric', 'ultralytics', 'onnxruntime')]\n"
        "if bad_modules:\n"
        "    sys.exit(f'FAIL: {bad_modules}')\n"
        "sys.exit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Heavy imports detected: {result.stderr or result.stdout}"


# ---------------------------------------------------------------------------
# GET /health Tests
# ---------------------------------------------------------------------------


def test_get_health(client):
    """GET /health returns 200, valid ISO timestamp, and version."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    model = HealthStatus(**data)
    assert model.status == "healthy"
    assert model.version == "1.0.0"

    parsed_time = datetime.fromisoformat(model.timestamp)
    assert parsed_time is not None


# ---------------------------------------------------------------------------
# GET /health/live Tests
# ---------------------------------------------------------------------------


def test_get_health_live(client):
    """GET /health/live returns 200 and status alive."""
    response = client.get("/health/live")
    assert response.status_code == 200

    data = response.json()
    model = LiveStatus(**data)
    assert model.status == "alive"


# ---------------------------------------------------------------------------
# GET /health/ready Tests
# ---------------------------------------------------------------------------


def test_get_health_ready_default(client):
    """GET /health/ready returns 200, status ready, model_loaded False, and device."""
    response = client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    model = ReadyStatus(**data)
    assert model.status == "ready"
    assert model.model_loaded is False
    assert isinstance(model.device, str)
    assert len(model.device) > 0


def test_get_health_ready_model_loaded_state(app, client):
    """GET /health/ready reflects app.state.model_loaded when True."""
    app.state.model_loaded = True
    response = client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    model = ReadyStatus(**data)
    assert model.model_loaded is True


def test_get_health_ready_detects_pipeline_model(app, client):
    """GET /health/ready detects when model or pipeline is configured in app.state."""
    fake_pipeline = MagicMock()
    fake_pipeline.inference_runtime = MagicMock()
    app.state.model_loaded = None
    app.state.pipeline = fake_pipeline

    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_get_health_ready_device_override_state(app, client):
    """GET /health/ready respects explicit app.state.device."""
    app.state.device = "cuda:0"
    response = client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["device"] == "cuda:0"


def test_get_health_ready_device_from_env(monkeypatch, client):
    """GET /health/ready respects CROWDDNA_DEVICE environment variable."""
    monkeypatch.setenv("CROWDDNA_DEVICE", "custom_npu")
    response = client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["device"] == "custom_npu"


def test_get_health_ready_does_not_load_models_eagerly(app, client):
    """Readiness probe must not eagerly load any model artifacts."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert getattr(app.state, "pipeline", None) is None


# ---------------------------------------------------------------------------
# GET /version Tests
# ---------------------------------------------------------------------------


def test_get_version(client):
    """GET /version returns api_version and engine_version."""
    response = client.get("/version")
    assert response.status_code == 200

    data = response.json()
    model = VersionInfo(**data)
    assert model.api_version == "1.0.0"
    assert model.engine_version == "1.0.0"


# ---------------------------------------------------------------------------
# OpenAPI & Docs Generation Tests
# ---------------------------------------------------------------------------


def test_openapi_json(client):
    """GET /openapi.json generates valid OpenAPI 3 schema with all endpoints and models."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert data["openapi"].startswith("3.")
    assert data["info"]["title"] == "CrowdDNA API"
    assert data["info"]["version"] == "1.0.0"

    paths = data["paths"]
    assert "/health" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/version" in paths

    schemas = data["components"]["schemas"]
    for schema_name in (
        "HealthStatus",
        "LiveStatus",
        "ReadyStatus",
        "VersionInfo",
        "ErrorResponse",
        "ErrorDetail",
    ):
        assert (
            schema_name in schemas
        ), f"Schema {schema_name} missing from OpenAPI components"


def test_docs_and_redoc_endpoints(client):
    """Swagger UI and ReDoc documentation endpoints are accessible."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "text/html" in docs_resp.headers["content-type"]

    redoc_resp = client.get("/redoc")
    assert redoc_resp.status_code == 200
    assert "text/html" in redoc_resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Universal Error Handling Tests
# ---------------------------------------------------------------------------


def test_error_404_not_found(client):
    """Accessing non-existent route returns 404 ErrorResponse."""
    response = client.get("/api/v1/missing_endpoint")
    assert response.status_code == 404

    data = response.json()
    model = ErrorResponse(**data)
    assert model.status_code == 404
    assert model.error == "Not Found"
    assert model.detail == "Not Found"


def test_error_405_method_not_allowed(client):
    """Calling existing route with wrong HTTP method returns 405 ErrorResponse."""
    response = client.post("/health")
    assert response.status_code == 405

    data = response.json()
    model = ErrorResponse(**data)
    assert model.status_code == 405
    assert model.error == "Method Not Allowed"


def test_error_422_request_validation(app, client):
    """Request parameter validation failures return 422 ErrorResponse with ErrorDetails."""

    @app.get("/test/param-validation")
    def sample_endpoint(param_int: int):
        return {"param_int": param_int}

    response = client.get("/test/param-validation?param_int=not_a_number")
    assert response.status_code == 422

    data = response.json()
    model = ErrorResponse(**data)
    assert model.status_code == 422
    assert model.error == "Validation Error"
    assert isinstance(model.detail, list)
    assert len(model.detail) > 0
    assert isinstance(model.detail[0], ErrorDetail)
    assert "param_int" in model.detail[0].loc


@pytest.mark.parametrize(
    "exception_cls,status_code,phrase",
    [
        (UploadSizeExceededError, 413, "Request Entity Too Large"),
        (InvalidVideoFormatError, 400, "Bad Request"),
        (VideoCorruptionError, 400, "Bad Request"),
        (ModelInferenceError, 500, "Internal Server Error"),
        (CrowdFlowError, 400, "Bad Request"),
    ],
)
def test_error_crowdflow_domain_exceptions(
    app, client, exception_cls, status_code, phrase
):
    """CrowdFlow domain exceptions are mapped to standardized HTTP ErrorResponse."""
    route_path = f"/test/domain-error-{status_code}-{exception_cls.__name__}"

    @app.get(route_path)
    def domain_error_endpoint():
        raise exception_cls(f"Simulated error for {exception_cls.__name__}")

    response = client.get(route_path)
    assert response.status_code == status_code

    data = response.json()
    model = ErrorResponse(**data)
    assert model.status_code == status_code
    assert model.error == phrase
    assert exception_cls.__name__ in str(model.detail)


def test_error_500_unhandled_exception(app, client):
    """Unhandled internal server exceptions return 500 ErrorResponse."""

    @app.get("/test/crash")
    def crashing_endpoint():
        raise RuntimeError("Unexpected internal crash")

    response = client.get("/test/crash")
    assert response.status_code == 500

    data = response.json()
    model = ErrorResponse(**data)
    assert model.status_code == 500
    assert model.error == "Internal Server Error"
    assert model.detail == "Internal server error"


# ---------------------------------------------------------------------------
# Schema Model Unit Tests
# ---------------------------------------------------------------------------


def test_schema_model_instantiations():
    """Verify direct instantiation and serialization of schema models."""
    health = HealthStatus(
        status="healthy", timestamp="2026-01-01T00:00:00Z", version="1.0.0"
    )
    assert health.model_dump()["status"] == "healthy"

    live = LiveStatus()
    assert live.model_dump()["status"] == "alive"

    ready = ReadyStatus(status="ready", model_loaded=False, device="cpu")
    assert ready.model_dump()["device"] == "cpu"

    version = VersionInfo(api_version="1.0.0", engine_version="1.0.0")
    assert version.model_dump()["api_version"] == "1.0.0"

    detail = ErrorDetail(loc=["body", "video"], msg="Required field", type="missing")
    dumped_detail = detail.model_dump()
    assert dumped_detail["loc"] == ["body", "video"]
    assert dumped_detail["msg"] == "Required field"

    error_resp = ErrorResponse(
        detail=[detail],
        error="Validation Error",
        status_code=422,
    )
    dumped_err = error_resp.model_dump()
    assert dumped_err["status_code"] == 422
    assert isinstance(dumped_err["detail"], list)
