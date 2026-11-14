"""Exception hierarchy for the CrowdDNA SDK."""

from __future__ import annotations

import builtins
from typing import Any


class CrowdDNAError(Exception):
    """Base exception class for all CrowdDNA SDK errors."""

    def __init__(self, message: str = "An error occurred in CrowdDNA SDK") -> None:
        super().__init__(message)
        self.message = message


class APIError(CrowdDNAError):
    """Raised when an API request fails with an HTTP error status code.

    Attributes:
        message: Error description.
        status_code: HTTP response status code.
        detail: Detailed error message or payload from the server.
        response_body: Raw or parsed response body if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Any = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.response_body = response_body

    def __str__(self) -> str:
        if self.detail:
            return f"[{self.status_code}] {self.message}: {self.detail}"
        return f"[{self.status_code}] {self.message}"


class ValidationError(APIError):
    """Raised when request validation fails (HTTP 400 or 422)."""

    def __init__(
        self,
        message: str = "Validation failed",
        status_code: int = 422,
        detail: Any = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            detail=detail,
            response_body=response_body,
        )


class NotFoundError(APIError):
    """Raised when a requested resource is not found (HTTP 404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        detail: Any = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            detail=detail,
            response_body=response_body,
        )


class JobCancelledError(CrowdDNAError):
    """Raised when an asynchronous job has been cancelled.

    Attributes:
        message: Description of the cancellation.
        job_id: ID of the cancelled job.
    """

    def __init__(
        self,
        message: str = "Job was cancelled",
        job_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.job_id = job_id


class ConnectionError(CrowdDNAError, builtins.ConnectionError):
    """Raised when a network connection to the CrowdDNA API server fails."""

    def __init__(
        self, message: str = "Failed to connect to CrowdDNA API server"
    ) -> None:
        super().__init__(message)


class TimeoutError(CrowdDNAError, builtins.TimeoutError):
    """Raised when an API request or job polling operation times out."""

    def __init__(self, message: str = "Operation timed out") -> None:
        super().__init__(message)
