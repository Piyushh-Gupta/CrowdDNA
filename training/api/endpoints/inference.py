from typing import Any
from training.api.routing import Router
from training.api.services.inference import InferenceService

def inference_handler(request: Any, context: Any) -> Any:
    service = InferenceService()
    return service.execute_pipeline(request, context)

Router.add_route("/api/v1/inference", "POST", inference_handler)
