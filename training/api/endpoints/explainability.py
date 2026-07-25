from typing import Any
from training.api.routing import Router
from training.api.services.explainability import ExplainabilityService

def explainability_handler(request: Any, context: Any) -> Any:
    service = ExplainabilityService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/explainability", "GET", explainability_handler)
