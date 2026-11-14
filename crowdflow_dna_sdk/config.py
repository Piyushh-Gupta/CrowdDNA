"""Configuration module for the CrowdDNA SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientConfig:
    """Configuration options for the CrowdDNA client.

    Attributes:
        base_url: Base URL of the CrowdDNA API server.
        timeout: Timeout in seconds for HTTP requests.
        headers: Default HTTP headers included with all requests.
        api_key: Optional API key for bearer authentication.
    """

    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Normalize base URL and populate authentication headers."""
        self.base_url = self.base_url.rstrip("/")
        if self.api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary containing configuration options.
        """
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "headers": dict(self.headers),
            "api_key": self.api_key,
        }
