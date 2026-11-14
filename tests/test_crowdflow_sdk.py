"""Comprehensive tests for crowdflow_dna_sdk."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

# Ensure CrowdDNA-API root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from crowdflow_dna_sdk import (  # noqa: E402
    APIError,
    ClientConfig,
    ConnectionError,
    CrowdDNAClient,
    CrowdDNAError,
    HealthStatus,
    InferenceResult,
    JobCancelledError,
    JobResult,
    JobStatus,
    LiveStatus,
    NotFoundError,
    ReadyStatus,
    RiskPrediction,
    TimeoutError,
    TimelineEntry,
    ValidationError,
    VersionInfo,
    VideoMetadata,
)


# =============================================================================
# Helper Utilities for Mocking HTTP Responses
# =============================================================================


def _create_mock_response(status_code: int = 200, payload: dict | str = ""):
    """Create a mock HTTP response compatible with urllib.request.urlopen."""
    if isinstance(payload, dict):
        body_bytes = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        body_bytes = payload.encode("utf-8")
    else:
        body_bytes = b""

    mock_resp = MagicMock()
    mock_resp.status = status_code
    mock_resp.read.return_value = body_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def _create_http_error(status_code: int, payload: dict | str = ""):
    """Create a urllib.error.HTTPError with given status code and body."""
    if isinstance(payload, dict):
        body_bytes = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        body_bytes = payload.encode("utf-8")
    else:
        body_bytes = b""

    fp = io.BytesIO(body_bytes)
    return urllib.error.HTTPError(
        url="http://localhost:8000/test",
        code=status_code,
        msg=f"HTTP {status_code}",
        hdrs={},
        fp=fp,
    )


# =============================================================================
# 1. ClientConfig Tests
# =============================================================================


def test_client_config_defaults():
    """Verify default ClientConfig attributes."""
    config = ClientConfig()
    assert config.base_url == "http://localhost:8000"
    assert config.timeout == 30.0
    assert config.headers == {}
    assert config.api_key is None


def test_client_config_custom_values():
    """Verify ClientConfig respects custom parameters and normalizes base_url."""
    config = ClientConfig(
        base_url="https://api.crowddna.io/v1/",
        timeout=15.0,
        headers={"X-Custom": "val"},
        api_key="secret-key-123",
    )
    assert config.base_url == "https://api.crowddna.io/v1"
    assert config.timeout == 15.0
    assert config.headers["X-Custom"] == "val"
    assert config.headers["Authorization"] == "Bearer secret-key-123"

    cfg_dict = config.to_dict()
    assert cfg_dict["base_url"] == "https://api.crowddna.io/v1"
    assert cfg_dict["timeout"] == 15.0
    assert cfg_dict["api_key"] == "secret-key-123"


# =============================================================================
# 2. Exceptions Tests
# =============================================================================


def test_exception_hierarchy():
    """Verify SDK exception class inheritance."""
    assert issubclass(APIError, CrowdDNAError)
    assert issubclass(ValidationError, APIError)
    assert issubclass(NotFoundError, APIError)
    assert issubclass(JobCancelledError, CrowdDNAError)
    assert issubclass(ConnectionError, CrowdDNAError)
    assert issubclass(ConnectionError, builtins.ConnectionError)
    assert issubclass(TimeoutError, CrowdDNAError)
    assert issubclass(TimeoutError, builtins.TimeoutError)


def test_api_error_attributes_and_str():
    """Verify APIError attributes and string representation."""
    err = APIError(
        message="Internal server error",
        status_code=500,
        detail="Pipeline crashed",
        response_body={"error": "fatal"},
    )
    assert err.status_code == 500
    assert err.message == "Internal server error"
    assert err.detail == "Pipeline crashed"
    assert err.response_body == {"error": "fatal"}
    assert "[500] Internal server error: Pipeline crashed" in str(err)

    err_no_detail = APIError(message="Forbidden", status_code=403)
    assert str(err_no_detail) == "[403] Forbidden"


def test_job_cancelled_error():
    """Verify JobCancelledError attributes."""
    err = JobCancelledError("Job was cancelled by operator", job_id="job-999")
    assert err.job_id == "job-999"
    assert "Job was cancelled by operator" in str(err)


# =============================================================================
# 3. Models Serialization & Conversion Tests
# =============================================================================


def test_risk_prediction_model():
    """Verify RiskPrediction instantiation, serialization, and deserialization."""
    pred = RiskPrediction(region_id=1, label="Critical", confidence=0.88)
    assert pred.region_id == 1
    assert pred.label == "Critical"
    assert pred.confidence == 0.88

    d = pred.to_dict()
    assert d == {"region_id": 1, "label": "Critical", "confidence": 0.88}

    pred_parsed = RiskPrediction.from_dict(
        {"region_id": "2", "label": "Safe", "confidence": "0.15"}
    )
    assert pred_parsed.region_id == 2
    assert pred_parsed.label == "Safe"
    assert pred_parsed.confidence == 0.15


def test_timeline_entry_model():
    """Verify TimelineEntry serialization and nested risk prediction parsing."""
    raw = {
        "frame_index": 5,
        "predictions": [
            {"region_id": 10, "label": "Congesting", "confidence": 0.72},
            {"region_id": 12, "label": "Safe", "confidence": 0.11},
        ],
    }
    entry = TimelineEntry.from_dict(raw)
    assert entry.frame_index == 5
    assert len(entry.predictions) == 2
    assert entry.predictions[0].region_id == 10
    assert entry.predictions[0].label == "Congesting"

    serialized = entry.to_dict()
    assert serialized["frame_index"] == 5
    assert len(serialized["predictions"]) == 2
    assert serialized["predictions"][0]["region_id"] == 10


def test_video_metadata_model():
    """Verify VideoMetadata attributes and extra field retention."""
    data = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 900,
        "duration_seconds": 30.0,
        "sample_rate": 2,
        "codec": "h264",
        "bitrate_kbps": 4500,
    }
    meta = VideoMetadata.from_dict(data)
    assert meta.fps == 30.0
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.duration_seconds == 30.0
    assert meta.extra["codec"] == "h264"

    meta_dict = meta.to_dict()
    assert meta_dict["codec"] == "h264"
    assert meta_dict["width"] == 1920


def test_inference_result_and_job_result_alias():
    """Verify InferenceResult model parsing, properties, and JobResult alias."""
    assert JobResult is InferenceResult

    data = {
        "timeline": [
            {
                "frame_index": 0,
                "predictions": [{"region_id": 1, "label": "Safe", "confidence": 0.2}],
            }
        ],
        "metadata": {"fps": 25.0, "width": 1280, "height": 720, "frame_count": 100},
        "output_video_url": "https://cdn.crowddna.io/runs/run1.mp4",
        "predicted_class": 1,
        "probabilities": [0.1, 0.85, 0.05],
        "confidence": 0.85,
        "backend": "onnxruntime",
        "inference_time_ms": 12.4,
        "model_format": "onnx",
        "model_version": "v1.2.0",
    }
    res = InferenceResult.from_dict(data)
    assert len(res.timeline) == 1
    assert res.timeline[0].frame_index == 0
    assert res.metadata is not None
    assert res.metadata.fps == 25.0
    assert res.output_video_url == "https://cdn.crowddna.io/runs/run1.mp4"
    assert res.confidence == 0.85
    assert res.backend == "onnxruntime"

    d = res.to_dict()
    assert d["output_video_url"] == "https://cdn.crowddna.io/runs/run1.mp4"
    assert d["probabilities"] == [0.1, 0.85, 0.05]


def test_job_status_model_and_properties():
    """Verify JobStatus state properties and nested result handling."""
    job_pending = JobStatus(job_id="job-1", status="PENDING")
    assert job_pending.is_running is True
    assert job_pending.is_completed is False
    assert job_pending.is_failed is False
    assert job_pending.is_cancelled is False

    job_completed = JobStatus(job_id="job-2", status="COMPLETED", progress=1.0)
    assert job_completed.is_running is False
    assert job_completed.is_completed is True

    job_failed = JobStatus(job_id="job-3", status="FAILED", error="Memory error")
    assert job_failed.is_failed is True
    assert job_failed.error == "Memory error"

    job_cancelled = JobStatus(job_id="job-4", status="CANCELLED")
    assert job_cancelled.is_cancelled is True

    # from_dict with nested result
    data = {
        "job_id": "job-5",
        "status": "COMPLETED",
        "progress": 1.0,
        "result": {
            "predicted_class": 0,
            "confidence": 0.95,
        },
    }
    status = JobStatus.from_dict(data)
    assert status.job_id == "job-5"
    assert status.result is not None
    assert status.result.confidence == 0.95
    assert status.to_dict()["job_id"] == "job-5"


def test_health_live_ready_version_models():
    """Verify HealthStatus, LiveStatus, ReadyStatus, and VersionInfo models."""
    health = HealthStatus.from_dict(
        {
            "status": "ok",
            "startup": True,
            "readiness": True,
            "liveness": True,
            "gpu": True,
        }
    )
    assert health.status == "ok"
    assert health.startup is True
    assert health.details["gpu"] is True
    assert health.to_dict()["gpu"] is True

    # LiveStatus dict-like interface
    live = LiveStatus.from_dict({"alive": True, "status": "ok", "uptime_s": 120})
    assert live.alive is True
    assert live["alive"] is True
    assert live["status"] == "ok"
    assert live.get("uptime_s") == 120
    assert "alive" in dict(live)

    # ReadyStatus dict-like interface
    ready = ReadyStatus.from_dict({"ready": True, "status": "ok", "models_loaded": 2})
    assert ready.ready is True
    assert ready["ready"] is True
    assert ready.get("models_loaded") == 2

    # VersionInfo
    version = VersionInfo.from_dict(
        {
            "version": "1.3.0",
            "api_version": "v1",
            "git_commit": "a1b2c3d",
            "build_date": "2026-11-14",
        }
    )
    assert version.version == "1.3.0"
    assert version.api_version == "v1"
    assert version.git_commit == "a1b2c3d"
    assert version.to_dict()["git_commit"] == "a1b2c3d"


# =============================================================================
# 4. CrowdDNAClient Initialization Tests
# =============================================================================


def test_client_init_defaults():
    """Verify CrowdDNAClient initialized with default configuration."""
    client = CrowdDNAClient()
    assert client.config.base_url == "http://localhost:8000"
    assert client.config.timeout == 30.0


def test_client_init_with_config():
    """Verify CrowdDNAClient initialized with explicit ClientConfig."""
    cfg = ClientConfig(
        base_url="https://api.crowddna.io", timeout=45.0, api_key="token-abc"
    )
    client = CrowdDNAClient(config=cfg)
    assert client.config.base_url == "https://api.crowddna.io"
    assert client.config.timeout == 45.0
    assert client.config.headers["Authorization"] == "Bearer token-abc"


def test_client_init_with_kwargs_override():
    """Verify kwargs override ClientConfig values."""
    cfg = ClientConfig(base_url="https://api.crowddna.io", timeout=45.0)
    client = CrowdDNAClient(
        config=cfg, base_url="http://override:9000", timeout=10.0, api_key="new-key"
    )
    assert client.config.base_url == "http://override:9000"
    assert client.config.timeout == 10.0
    assert client.config.headers["Authorization"] == "Bearer new-key"


# =============================================================================
# 5. Client Public Methods Unit Tests with Mocked Transport
# =============================================================================


@patch("urllib.request.urlopen")
def test_client_get_health(mock_urlopen):
    """Test get_health() returns properly typed HealthStatus."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {"status": "healthy", "startup": True, "readiness": True, "liveness": True},
    )

    client = CrowdDNAClient()
    health = client.get_health()

    assert isinstance(health, HealthStatus)
    assert health.status == "healthy"
    assert health.readiness is True


@patch("urllib.request.urlopen")
def test_client_get_liveness(mock_urlopen):
    """Test get_liveness() returns LiveStatus supporting dict operations."""
    mock_urlopen.return_value = _create_mock_response(
        200, {"alive": True, "status": "ok"}
    )

    client = CrowdDNAClient()
    live = client.get_liveness()

    assert isinstance(live, LiveStatus)
    assert live.alive is True
    assert live["alive"] is True
    assert live.get("status") == "ok"


@patch("urllib.request.urlopen")
def test_client_get_readiness(mock_urlopen):
    """Test get_readiness() returns ReadyStatus supporting dict operations."""
    mock_urlopen.return_value = _create_mock_response(
        200, {"ready": True, "status": "ok"}
    )

    client = CrowdDNAClient()
    ready = client.get_readiness()

    assert isinstance(ready, ReadyStatus)
    assert ready.ready is True
    assert ready["ready"] is True
    assert ready.get("status") == "ok"


@patch("urllib.request.urlopen")
def test_client_get_version(mock_urlopen):
    """Test get_version() returns VersionInfo."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {"version": "2.0.0", "api_version": "v1", "git_commit": "abcdef1"},
    )

    client = CrowdDNAClient()
    version = client.get_version()

    assert isinstance(version, VersionInfo)
    assert version.version == "2.0.0"
    assert version.git_commit == "abcdef1"


@patch("urllib.request.urlopen")
def test_client_submit_inference_with_file(mock_urlopen):
    """Test submit_inference() with a real temporary video file."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {"job_id": "job-101", "status": "PENDING", "progress": 0.0},
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"fake-video-bytes")
        tf_path = tf.name

    client = CrowdDNAClient()
    job = client.submit_inference(tf_path, enable_calibration=True)

    assert isinstance(job, JobStatus)
    assert job.job_id == "job-101"
    assert job.status == "PENDING"

    # Verify request payload was multipart
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.method == "POST"
    assert req.headers["Content-type"].startswith("multipart/form-data; boundary=")
    assert b"fake-video-bytes" in req.data
    assert b"enable_calibration" in req.data


@patch("urllib.request.urlopen")
def test_client_submit_inference_non_existent_path(mock_urlopen):
    """Test submit_inference() handles non-existent video path safely without crash."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {"job_id": "job-dummy", "status": "PENDING", "progress": 0.0},
    )

    client = CrowdDNAClient()
    job = client.submit_inference("non_existent_video.mp4", enable_calibration=False)

    assert job.job_id == "job-dummy"
    assert job.status == "PENDING"


@patch("urllib.request.urlopen")
def test_client_get_job(mock_urlopen):
    """Test get_job() retrieves JobStatus and parses result."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {
            "job_id": "job-abc",
            "status": "COMPLETED",
            "progress": 1.0,
            "result": {
                "output_video_url": "http://cdn/result.mp4",
                "predicted_class": 2,
                "confidence": 0.91,
            },
        },
    )

    client = CrowdDNAClient()
    job = client.get_job("job-abc")

    assert job.job_id == "job-abc"
    assert job.is_completed is True
    assert job.result is not None
    assert job.result.predicted_class == 2
    assert job.result.confidence == 0.91


@patch("urllib.request.urlopen")
def test_client_cancel_job(mock_urlopen):
    """Test cancel_job() issues DELETE request and returns cancelled status."""
    mock_urlopen.return_value = _create_mock_response(
        200,
        {"job_id": "job-to-cancel", "status": "CANCELLED", "progress": 0.5},
    )

    client = CrowdDNAClient()
    job = client.cancel_job("job-to-cancel")

    assert job.job_id == "job-to-cancel"
    assert job.is_cancelled is True

    req = mock_urlopen.call_args[0][0]
    assert req.method == "DELETE"


@patch("urllib.request.urlopen")
def test_client_cancel_job_empty_204_response(mock_urlopen):
    """Test cancel_job() handles 204 No Content response gracefully."""
    mock_urlopen.return_value = _create_mock_response(204, "")

    client = CrowdDNAClient()
    job = client.cancel_job("job-204")

    assert job.job_id == "job-204"
    assert job.is_cancelled is True


# =============================================================================
# 6. Polling & wait_for_job Tests
# =============================================================================


def test_wait_for_job_immediate_completion():
    """Test wait_for_job returns immediately when job is already completed."""
    client = CrowdDNAClient()
    completed_job = JobStatus(job_id="job-fast", status="COMPLETED", progress=1.0)
    with patch.object(client, "get_job", return_value=completed_job):
        res = client.wait_for_job(
            "job-fast", poll_interval_seconds=0.01, timeout_seconds=1.0
        )
        assert res.is_completed is True


def test_wait_for_job_transitions():
    """Test wait_for_job polls until state transitions to COMPLETED."""
    client = CrowdDNAClient()
    states = [
        JobStatus(job_id="job-seq", status="PENDING", progress=0.1),
        JobStatus(job_id="job-seq", status="PROCESSING", progress=0.6),
        JobStatus(job_id="job-seq", status="COMPLETED", progress=1.0),
    ]

    with patch.object(client, "get_job", side_effect=states) as mock_get_job:
        res = client.wait_for_job(
            "job-seq", poll_interval_seconds=0.01, timeout_seconds=2.0
        )
        assert res.is_completed is True
        assert mock_get_job.call_count == 3


def test_wait_for_job_cancelled_raises_error():
    """Test wait_for_job raises JobCancelledError when job is cancelled."""
    client = CrowdDNAClient()
    states = [
        JobStatus(job_id="job-cancel", status="PROCESSING", progress=0.3),
        JobStatus(job_id="job-cancel", status="CANCELLED"),
    ]

    with patch.object(client, "get_job", side_effect=states):
        with pytest.raises(JobCancelledError) as exc_info:
            client.wait_for_job(
                "job-cancel", poll_interval_seconds=0.01, timeout_seconds=1.0
            )
        assert exc_info.value.job_id == "job-cancel"


def test_wait_for_job_failed_raises_api_error():
    """Test wait_for_job raises APIError when job fails."""
    client = CrowdDNAClient()
    failed_job = JobStatus(
        job_id="job-fail", status="FAILED", error="Out of GPU memory"
    )

    with patch.object(client, "get_job", return_value=failed_job):
        with pytest.raises(APIError) as exc_info:
            client.wait_for_job(
                "job-fail", poll_interval_seconds=0.01, timeout_seconds=1.0
            )
        assert "Out of GPU memory" in str(exc_info.value)
        assert exc_info.value.status_code == 500


def test_wait_for_job_timeout_raises_timeout_error():
    """Test wait_for_job raises TimeoutError if deadline is exceeded."""
    client = CrowdDNAClient()
    pending_job = JobStatus(job_id="job-slow", status="PROCESSING", progress=0.2)

    with patch.object(client, "get_job", return_value=pending_job):
        with pytest.raises(TimeoutError) as exc_info:
            client.wait_for_job(
                "job-slow", poll_interval_seconds=0.02, timeout_seconds=0.05
            )
        assert "timed out after 0.1 seconds" in str(
            exc_info.value
        ) or "timed out" in str(exc_info.value)


# =============================================================================
# 7. HTTP Error Code Mapping Tests
# =============================================================================


@patch("urllib.request.urlopen")
def test_error_mapping_400_validation_error(mock_urlopen):
    """Test HTTP 400 Bad Request maps to ValidationError."""
    problem_details = {
        "title": "Invalid Video Format",
        "detail": "The uploaded file is not a supported MP4 or AVI format.",
        "status": 400,
    }
    mock_urlopen.side_effect = _create_http_error(400, problem_details)

    client = CrowdDNAClient()
    with pytest.raises(ValidationError) as exc_info:
        client.get_health()

    err = exc_info.value
    assert err.status_code == 400
    assert "Invalid Video Format" in err.message or "supported MP4" in err.message
    assert err.detail == "The uploaded file is not a supported MP4 or AVI format."


@patch("urllib.request.urlopen")
def test_error_mapping_422_unprocessable_entity(mock_urlopen):
    """Test HTTP 422 Unprocessable Entity maps to ValidationError."""
    error_payload = {
        "detail": [
            {
                "loc": ["body", "file"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }
    mock_urlopen.side_effect = _create_http_error(422, error_payload)

    client = CrowdDNAClient()
    with pytest.raises(ValidationError) as exc_info:
        client.get_health()

    err = exc_info.value
    assert err.status_code == 422
    assert isinstance(err.detail, list)


@patch("urllib.request.urlopen")
def test_error_mapping_404_not_found(mock_urlopen):
    """Test HTTP 404 Not Found maps to NotFoundError."""
    mock_urlopen.side_effect = lambda *args, **kwargs: (_ for _ in ()).throw(
        _create_http_error(404, {"detail": "Job not found"})
    )

    client = CrowdDNAClient()
    with pytest.raises(NotFoundError) as exc_info:
        client.get_job("non-existent-id")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Job not found"


@patch("urllib.request.urlopen")
def test_error_mapping_500_api_error(mock_urlopen):
    """Test HTTP 500 Internal Server Error maps to APIError."""
    mock_urlopen.side_effect = _create_http_error(
        500, {"message": "Crash in inference engine"}
    )

    client = CrowdDNAClient()
    with pytest.raises(APIError) as exc_info:
        client.get_health()

    assert exc_info.value.status_code == 500
    assert "Crash in inference engine" in exc_info.value.message


@patch("urllib.request.urlopen")
def test_error_mapping_504_gateway_timeout(mock_urlopen):
    """Test HTTP 504 Gateway Timeout maps to TimeoutError."""
    mock_urlopen.side_effect = _create_http_error(504, "Gateway Timeout")

    client = CrowdDNAClient()
    with pytest.raises(TimeoutError) as exc_info:
        client.get_health()

    assert "Request timed out" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_error_mapping_connection_refused(mock_urlopen):
    """Test network connection failure maps to ConnectionError."""
    mock_urlopen.side_effect = urllib.error.URLError(reason="Connection refused")

    client = CrowdDNAClient()
    with pytest.raises(ConnectionError) as exc_info:
        client.get_health()

    assert "Connection refused" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_error_mapping_socket_timeout(mock_urlopen):
    """Test socket timeout maps to TimeoutError."""
    mock_urlopen.side_effect = socket.timeout("timed out")

    client = CrowdDNAClient()
    with pytest.raises(TimeoutError) as exc_info:
        client.get_health()

    assert "Request timed out" in str(exc_info.value)


# =============================================================================
# 8. Fallback Routing Tests
# =============================================================================


@patch("urllib.request.urlopen")
def test_endpoint_fallback_on_404(mock_urlopen):
    """Test client falls back from primary path to fallback_path on 404."""
    # First call to /health fails with 404, second call to /api/v1/health succeeds with 200
    mock_urlopen.side_effect = [
        _create_http_error(404, "Not Found"),
        _create_mock_response(
            200, {"status": "ok", "startup": True, "readiness": True, "liveness": True}
        ),
    ]

    client = CrowdDNAClient()
    health = client.get_health()

    assert health.status == "ok"
    assert mock_urlopen.call_count == 2
    # Verify first URL was /health, second was /api/v1/health
    call1_req = mock_urlopen.call_args_list[0][0][0]
    call2_req = mock_urlopen.call_args_list[1][0][0]
    assert call1_req.full_url.endswith("/health")
    assert call2_req.full_url.endswith("/api/v1/health")


@patch("urllib.request.urlopen")
def test_malformed_json_response_raises_api_error(mock_urlopen):
    """Test that a 200 OK response with non-JSON text raises APIError."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = (
        b"<html><head><title>502 Bad Gateway</title></head></html>"
    )
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    client = CrowdDNAClient()
    with pytest.raises(APIError) as exc_info:
        client.get_health()

    assert exc_info.value.status_code == 200
    assert "Malformed response: expected JSON" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_malformed_http_error_body_handled(mock_urlopen):
    """Test that an HTTP error with HTML / raw text body is cleanly handled."""
    mock_urlopen.side_effect = _create_http_error(
        500, "Internal Server Error: Database Down"
    )

    client = CrowdDNAClient()
    with pytest.raises(APIError) as exc_info:
        client.get_job("some-uuid")

    assert exc_info.value.status_code == 500
    assert "Internal Server Error: Database Down" in str(exc_info.value)
