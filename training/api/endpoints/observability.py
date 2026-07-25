from typing import Any
from training.api.routing import Router
from training.api.services.observability import ObservabilityService

def observability_handler(request: Any, context: Any) -> Any:
    service = ObservabilityService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/observability", "GET", observability_handler)
