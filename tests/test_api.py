import pytest
from training.api.application import Application
from training.api.server import Server
from training.api.registry import EndpointRegistry
from training.api.responses import ResponseFactory
from training.api.exceptions import ValidationException

def test_application_startup():
    app = Application()
    server = Server(app)
    assert not app.started
    server.run()
    assert app.started

def test_route_registration():
    import training.api.endpoints.workflow  # noqa: F401
    import training.api.endpoints.health  # noqa: F401
    assert EndpointRegistry.get("POST /api/v1/workflow") is not None
    assert EndpointRegistry.get("GET /api/v1/health") is not None
    assert EndpointRegistry.get("GET /api/v1/missing") is None

def test_response_factory():
    resp = ResponseFactory.success({"status": "ok"}, "req-1", "corr-1")
    assert resp.success is True
    assert resp.data["status"] == "ok"
    assert resp.request_id == "req-1"

def test_exceptions():
    with pytest.raises(ValidationException) as exc:
        raise ValidationException("Invalid payload")
    
    assert exc.value.code == "VALIDATION_ERROR"
    assert exc.value.status_code == 400
