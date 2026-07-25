from typing import Any
from training.api.routing import Router
from training.api.services.workflow import WorkflowService

def workflow_handler(request: Any, context: Any) -> Any:
    service = WorkflowService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/workflow", "POST", workflow_handler)
