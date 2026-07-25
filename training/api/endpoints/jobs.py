from typing import Any
from training.api.routing import Router
from training.api.services.jobs import JobsService

def jobs_handler(request: Any, context: Any) -> Any:
    service = JobsService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/jobs", "GET", jobs_handler)
