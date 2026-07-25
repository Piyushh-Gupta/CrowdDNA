from typing import Any
from training.api.routing import Router
from training.api.services.experiments import ExperimentsService

def experiments_handler(request: Any, context: Any) -> Any:
    service = ExperimentsService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/experiments", "GET", experiments_handler)
