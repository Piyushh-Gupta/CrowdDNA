import time
import uuid
from typing import Callable, Any
from training.api.context import APIContext
from training.api.metadata import ApiMetadata
from training.api.authentication import ApiAuthentication
from training.api.exceptions import ApiException
from training.api.responses import ResponseFactory
from training.api.dto.responses import ApiError

class MiddlewarePipeline:
    @staticmethod
    def process_request(request: Any, handler: Callable) -> Any:
        # 1. Extract context variables
        headers = getattr(request, "headers", {})
        token = ApiAuthentication.extract_token(headers)
        identity = token if token else None # Mock identity resolution
        
        req_id = str(uuid.uuid4())
        corr_id = headers.get("x-correlation-id", str(uuid.uuid4()))
        idempotency_key = headers.get("idempotency-key")
        
        context = APIContext(
            request_id=req_id,
            correlation_id=corr_id,
            client_ip="127.0.0.1",
            user_agent="Unknown",
            request_start_timestamp=time.monotonic(), # Use monotonic for precise duration
            api_version=ApiMetadata().api_version,
            identity=identity,
            idempotency_key=idempotency_key
        )
        
        # 2. Execute handler with global exception mapping
        try:
            return handler(request, context)
        except ApiException as e:
            error = ApiError(code=e.code, message=e.message)
            return ResponseFactory.error((error,), req_id, corr_id)
        except Exception:
            error = ApiError(code="INTERNAL_ERROR", message="An unexpected error occurred.")
            # We preserve cause in logs, but return sanitized error
            return ResponseFactory.error((error,), req_id, corr_id)
