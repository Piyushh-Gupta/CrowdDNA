from typing import Any
from training.api.services.base import BaseService
from training.api.context import APIContext

class JobsService(BaseService):
    def execute(self, request: Any, context: APIContext) -> Any:
        return {"job_status": "running"}
