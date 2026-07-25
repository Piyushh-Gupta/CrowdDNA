# Coding Standards

## Python Conventions
- Follow **PEP 8** strictly, enforced via `ruff check .`.
- **Type Hints**: Mandatory for all function signatures and class properties. No `Any` without explicit justification.

## Design Patterns
- **Immutability**: Prefer frozen `dataclasses` (`@dataclass(frozen=True)`).
- **Configuration**: Hardcoded values are prohibited. Use central configuration frameworks (Phase 19).
- **Exceptions**: Use typed, domain-specific exceptions inheriting from a base module error. Never use bare `except:`.

## Concurrency
- **Thread Safety**: Shared mutable state must be guarded via `threading.Lock()`.
- Avoid global singleton mutations.

## Documentation & Logging
- **Logging**: Use structured JSON logging via the Observability framework.
- **Documentation**: All public APIs must have Google-style docstrings.

## Testing
- **Unit Tests**: Required for all pure logic.
- **Integration Tests**: Required for all API, Database, and File I/O operations.
