class SecurityValidationError(Exception):
    """Base exception for security validation."""
    pass

class RegistryConfigurationError(SecurityValidationError):
    """Raised when the registry is improperly configured."""
    pass
