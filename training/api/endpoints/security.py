from typing import Any
from training.api.routing import Router
from training.api.services.security import SecurityService

def security_handler(request: Any, context: Any) -> Any:
    service = SecurityService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/security", "GET", security_handler)
