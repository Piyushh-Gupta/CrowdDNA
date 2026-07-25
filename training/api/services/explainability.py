from typing import Any
from training.api.services.base import BaseService
from training.api.context import APIContext

class ExplainabilityService(BaseService):
    def execute(self, request: Any, context: APIContext) -> Any:
        return {"shap_values": []}
