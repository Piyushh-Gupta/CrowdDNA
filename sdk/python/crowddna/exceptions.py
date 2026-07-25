class SDKError(Exception):
    pass

class AuthenticationError(SDKError):
    pass

class AuthorizationError(SDKError):
    pass

class ValidationError(SDKError):
    pass

class ConflictError(SDKError):
    pass

class RateLimitError(SDKError):
    pass

class ServerError(SDKError):
    pass

class TransportError(SDKError):
    pass
