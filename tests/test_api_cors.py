"""Tests for Cross-Origin Resource Sharing (CORS) on CrowdDNA FastAPI endpoints.

Verifies that browser preflight (OPTIONS) requests and simple/credentialed HTTP requests
receive the required CORS response headers across system and inference routes.
"""

import pytest
from fastapi.testclient import TestClient

from crowdflow_dna.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient instance."""
    app = create_app()
    return TestClient(app)


def test_options_preflight_health(client: TestClient) -> None:
    """Verify OPTIONS preflight request to /health returns 200 and CORS headers."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/health", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [
        "http://localhost:5173",
        "*",
    ]
    assert "access-control-allow-methods" in response.headers
    assert "GET" in response.headers["access-control-allow-methods"]
    assert "access-control-allow-headers" in response.headers
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_options_preflight_inference(client: TestClient) -> None:
    """Verify OPTIONS preflight request to /inference returns 200 and CORS headers."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, Idempotency-Key",
    }
    response = client.options("/inference", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [
        "http://localhost:5173",
        "*",
    ]
    assert "access-control-allow-methods" in response.headers
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "access-control-allow-headers" in response.headers
    allow_headers = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allow_headers or "*" in allow_headers


def test_get_health_with_origin_header(client: TestClient) -> None:
    """Verify GET /health with an Origin header returns Access-Control-Allow-Origin."""
    headers = {"Origin": "http://localhost:5173"}
    response = client.get("/health", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [
        "http://localhost:5173",
        "*",
    ]


def test_options_preflight_inference_job(client: TestClient) -> None:
    """Verify OPTIONS preflight request to /inference/{job_id} returns 200 and CORS headers."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Authorization",
    }
    response = client.options("/inference/job-12345", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [
        "http://localhost:3000",
        "*",
    ]
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers


def test_options_preflight_delete_job(client: TestClient) -> None:
    """Verify OPTIONS preflight request for DELETE /inference/{job_id} returns 200 and CORS headers."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "DELETE",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/inference/job-12345", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "DELETE" in response.headers["access-control-allow-methods"]


def test_cors_headers_present_on_error(client: TestClient) -> None:
    """Verify that error responses still include CORS headers when Origin is provided."""
    headers = {"Origin": "http://localhost:5173"}
    response = client.get("/nonexistent-route", headers=headers)

    assert response.status_code == 404
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in [
        "http://localhost:5173",
        "*",
    ]


def test_cors_disallowed_origin(client: TestClient) -> None:
    """Verify that origins outside the configured whitelist do not receive CORS headers."""
    headers = {"Origin": "http://malicious-site.com"}
    response = client.get("/health", headers=headers)

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that CROWDDNA_CORS_ORIGINS configures custom origin whitelist."""
    monkeypatch.setenv(
        "CROWDDNA_CORS_ORIGINS",
        "https://crowddna.internal, https://analytics.company.com",
    )
    custom_app = create_app()
    custom_client = TestClient(custom_app)

    # Allowed custom origin
    res1 = custom_client.get(
        "/health", headers={"Origin": "https://crowddna.internal"}
    )
    assert res1.status_code == 200
    assert res1.headers.get("access-control-allow-origin") == "https://crowddna.internal"

    # Disallowed localhost (since custom whitelist replaced defaults)
    res2 = custom_client.get(
        "/health", headers={"Origin": "http://localhost:5173"}
    )
    assert res2.status_code == 200
    assert "access-control-allow-origin" not in res2.headers


def test_cors_wildcard_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that CROWDDNA_CORS_ORIGINS='*' allows arbitrary origins."""
    monkeypatch.setenv("CROWDDNA_CORS_ORIGINS", "*")
    wildcard_app = create_app()
    wildcard_client = TestClient(wildcard_app)

    res = wildcard_client.get(
        "/health", headers={"Origin": "https://anywhere.example.com"}
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "*"
