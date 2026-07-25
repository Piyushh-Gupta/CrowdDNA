import uuid
from typing import Any
from training.api.services.base import BaseService
from training.api.context import APIContext

class WorkflowService(BaseService):
    def execute(self, request: Any, context: APIContext) -> Any:
        # Mock delegating to Orchestrator
        job_id = str(uuid.uuid4())
        return {"job_id": job_id, "status": "accepted"}
