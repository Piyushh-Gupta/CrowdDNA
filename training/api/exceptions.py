class ApiException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ValidationException(ApiException):
    def __init__(self, message: str):
        super().__init__(code="VALIDATION_ERROR", message=message, status_code=400)

class AuthenticationException(ApiException):
    def __init__(self, message: str):
        super().__init__(code="UNAUTHORIZED", message=message, status_code=401)

class AuthorizationException(ApiException):
    def __init__(self, message: str):
        super().__init__(code="FORBIDDEN", message=message, status_code=403)

class ResourceNotFoundException(ApiException):
    def __init__(self, message: str):
        super().__init__(code="NOT_FOUND", message=message, status_code=404)

class IdempotencyConflictException(ApiException):
    def __init__(self, message: str):
        super().__init__(code="IDEMPOTENCY_CONFLICT", message=message, status_code=409)
