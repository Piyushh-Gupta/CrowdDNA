from typing import Any
from training.api.routing import Router
from training.api.services.reproducibility import ReproducibilityService

def reproducibility_handler(request: Any, context: Any) -> Any:
    service = ReproducibilityService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/reproducibility", "GET", reproducibility_handler)
