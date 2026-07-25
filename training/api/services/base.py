from typing import Any
import hashlib
import json
from dataclasses import is_dataclass, asdict
from training.api.context import APIContext
from training.api.exceptions import AuthorizationException, IdempotencyConflictException
from training.api.registry import IdempotencyRegistry

class BaseService:
    def _compute_fingerprint(self, request: Any) -> str:
        if is_dataclass(request):
            data = asdict(request)
        elif isinstance(request, dict):
            data = request
        else:
            data = str(request)
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def execute_pipeline(self, request: Any, context: APIContext) -> Any:
        self.authorize(request, context)
        self.validate(request, context)
        
        idempotency_key = context.idempotency_key
        cache_key = None
        fingerprint = None
        if idempotency_key:
            identity_str = context.identity if context.identity else "anonymous"
            cache_key = f"{identity_str}:{idempotency_key}"
            fingerprint = self._compute_fingerprint(request)
            cached_data = IdempotencyRegistry.get(cache_key)
            if cached_data is not None:
                cached_fingerprint, cached_result = cached_data
                if cached_fingerprint != fingerprint:
                    raise IdempotencyConflictException("Idempotency key reused with different payload.")
                return cached_result
                
        result = self.execute(request, context)
        self.observe(result, context)
        self.audit(result, context)
        
        if cache_key and fingerprint:
            IdempotencyRegistry.store(cache_key, fingerprint, result)
            
        return result

    def authorize(self, request: Any, context: APIContext):
        if not context.identity:
            raise AuthorizationException("Identity required")

    def validate(self, request: Any, context: APIContext):
        pass

    def execute(self, request: Any, context: APIContext) -> Any:
        raise NotImplementedError

    def observe(self, result: Any, context: APIContext):
        pass

    def audit(self, result: Any, context: APIContext):
        pass
