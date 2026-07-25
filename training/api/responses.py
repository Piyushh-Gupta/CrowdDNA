import time
from typing import Any, Tuple
from types import MappingProxyType
from training.api.dto.responses import ApiResponse, ApiError

class ResponseFactory:
    @staticmethod
    def success(data: Any, request_id: str, correlation_id: str) -> ApiResponse:
        return ApiResponse(
            success=True,
            data=data,
            errors=tuple(),
            metadata=MappingProxyType({}),
            request_id=request_id,
            correlation_id=correlation_id,
            timestamp=time.time()
        )

    @staticmethod
    def error(errors: Tuple[ApiError, ...], request_id: str, correlation_id: str) -> ApiResponse:
        return ApiResponse(
            success=False,
            data=None,
            errors=errors,
            metadata=MappingProxyType({}),
            request_id=request_id,
            correlation_id=correlation_id,
            timestamp=time.time()
        )
