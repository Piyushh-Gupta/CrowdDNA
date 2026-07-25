from typing import Any
from training.api.services.base import BaseService
from training.api.context import APIContext

class InferenceService(BaseService):
    def execute(self, request: Any, context: APIContext) -> Any:
        return {"prediction": [0.5, 0.5]}
