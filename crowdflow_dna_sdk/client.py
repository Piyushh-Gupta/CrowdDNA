"""Main client for the CrowdDNA REST API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Mapping

from crowdflow_dna_sdk.config import ClientConfig
from crowdflow_dna_sdk.exceptions import (
    APIError,
    ConnectionError,
    JobCancelledError,
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from crowdflow_dna_sdk.models import (
    HealthStatus,
    JobStatus,
    LiveStatus,
    ReadyStatus,
    VersionInfo,
)

logger = logging.getLogger(__name__)


class CrowdDNAClient:
    """Client for interacting with the CrowdDNA REST API.

    Provides high-level typed methods for querying system health, submitting
    asynchronous video inference jobs, and polling results.
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the CrowdDNA API client.

        Args:
            config: Optional pre-configured ClientConfig instance.
            base_url: Optional base URL overriding config.
            timeout: Optional request timeout in seconds overriding config.
            api_key: Optional API key for bearer authentication.
            headers: Optional default HTTP headers to merge.
        """
        if config is not None:
            self.config = ClientConfig(
                base_url=base_url if base_url is not None else config.base_url,
                timeout=timeout if timeout is not None else config.timeout,
                headers=dict(config.headers),
                api_key=api_key if api_key is not None else config.api_key,
            )
        else:
            self.config = ClientConfig(
                base_url=base_url or "http://localhost:8000",
                timeout=timeout if timeout is not None else 30.0,
                headers=dict(headers or {}),
                api_key=api_key,
            )

        if headers:
            self.config.headers.update(headers)

    def _build_multipart_payload(
        self,
        video_path: str,
        enable_calibration: bool,
    ) -> tuple[bytes, str]:
        """Construct multipart/form-data payload for video upload.

        Args:
            video_path: Path to the video file to upload.
            enable_calibration: Flag indicating whether to enable metric calibration.

        Returns:
            Tuple of (request_body_bytes, content_type_header).
        """
        boundary = f"----CrowdDNABoundary{uuid.uuid4().hex}"
        filename = Path(video_path).name or "video.mp4"

        if os.path.isfile(video_path):
            with open(video_path, "rb") as f:
                file_bytes = f.read()
        else:
            file_bytes = b""

        parts: list[bytes] = []

        # Video file part
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(
                "utf-8"
            )
        )
        parts.append(b"Content-Type: video/mp4\r\n\r\n")
        parts.append(file_bytes)
        parts.append(b"\r\n")

        # Enable calibration part
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            b'Content-Disposition: form-data; name="enable_calibration"\r\n\r\n'
        )
        parts.append(str(enable_calibration).lower().encode("utf-8"))
        parts.append(b"\r\n")

        # Closing boundary
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))

        body = b"".join(parts)
        content_type = f"multipart/form-data; boundary={boundary}"
        return body, content_type

    def _handle_http_error(self, status_code: int, raw_body: str) -> None:
        """Parse HTTP error response and raise appropriate custom exception.

        Args:
            status_code: HTTP response status code.
            raw_body: Raw string payload of error response.

        Raises:
            ValidationError: On 400 or 422 HTTP errors.
            NotFoundError: On 404 HTTP errors.
            TimeoutError: On 408 or 504 HTTP errors.
            APIError: On other 4xx/5xx HTTP errors.
        """
        detail: Any = None
        message = f"HTTP Error {status_code}"
        response_body: Any = raw_body

        if raw_body:
            try:
                parsed = json.loads(raw_body)
                response_body = parsed
                if isinstance(parsed, dict):
                    # RFC 7807 problem details or standard error schema
                    if "detail" in parsed:
                        detail = parsed["detail"]
                        message = detail if isinstance(detail, str) else str(detail)
                    elif "message" in parsed:
                        message = str(parsed["message"])
                        detail = parsed.get("details")
                    elif "title" in parsed:
                        message = str(parsed["title"])
                        detail = parsed.get("detail")
                    elif "error" in parsed:
                        message = str(parsed["error"])
                    elif (
                        "errors" in parsed
                        and isinstance(parsed["errors"], list)
                        and parsed["errors"]
                    ):
                        first_err = parsed["errors"][0]
                        if isinstance(first_err, dict) and "message" in first_err:
                            message = str(first_err["message"])
                        detail = parsed["errors"]
            except Exception:
                message = raw_body

        if status_code in (408, 504):
            raise TimeoutError(f"[{status_code}] Request timed out: {message}")
        if status_code in (400, 422):
            raise ValidationError(
                message=message,
                status_code=status_code,
                detail=detail,
                response_body=response_body,
            )
        if status_code == 404:
            raise NotFoundError(
                message=message,
                status_code=status_code,
                detail=detail,
                response_body=response_body,
            )
        raise APIError(
            message=message,
            status_code=status_code,
            detail=detail,
            response_body=response_body,
        )

    def _handle_network_error(self, exc: Exception) -> None:
        """Map low-level network exceptions to SDK connection/timeout errors.

        Args:
            exc: Low-level exception.

        Raises:
            TimeoutError: For connection and socket timeout conditions.
            ConnectionError: For socket and connection failures.
        """
        if isinstance(exc, (TimeoutError, socket.timeout)):
            raise TimeoutError(f"Request timed out: {exc}") from exc
        if isinstance(exc, urllib.error.URLError):
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise TimeoutError(f"Request timed out: {exc.reason}") from exc
            raise ConnectionError(f"Network connection failed: {exc.reason}") from exc
        raise ConnectionError(f"Network error: {exc}") from exc

    def _send_request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send HTTP request via urllib.request and return parsed JSON.

        Args:
            method: HTTP method string (e.g. "GET", "POST", "DELETE").
            path: Relative URL path starting with "/".
            body: Optional request body bytes.
            headers: Optional request-specific headers.

        Returns:
            Parsed JSON dictionary from the response.

        Raises:
            CrowdDNAError: On network or HTTP error.
        """
        url = f"{self.config.base_url}{path}"
        req_headers = dict(self.config.headers)
        if headers:
            req_headers.update(headers)

        if body is not None and "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url=url,
            data=body,
            headers=req_headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                raw_bytes = response.read()
                if not raw_bytes:
                    return {}
                resp_text = raw_bytes.decode("utf-8")
                try:
                    return json.loads(resp_text)
                except json.JSONDecodeError as err:
                    raise APIError(
                        message="Malformed response: expected JSON but got invalid content",
                        status_code=getattr(response, "status", 200),
                        response_body=resp_text,
                    ) from err
        except APIError:
            raise
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8") if exc.fp else ""
            self._handle_http_error(exc.code, raw_body)
            return {}  # Unreachable but satisfies type checker
        except Exception as exc:
            self._handle_network_error(exc)
            return {}  # Unreachable

    def _request_with_fallback(
        self,
        method: str,
        path: str,
        fallback_path: str | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute request with optional path fallback on 404 Not Found.

        Args:
            method: HTTP method.
            path: Primary endpoint path.
            fallback_path: Alternate endpoint path to attempt if primary returns 404.
            body: Optional body bytes.
            headers: Optional headers.

        Returns:
            Parsed JSON dictionary response.
        """
        try:
            return self._send_request(method, path, body=body, headers=headers)
        except NotFoundError:
            if fallback_path and fallback_path != path:
                return self._send_request(
                    method, fallback_path, body=body, headers=headers
                )
            raise

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    def get_health(self) -> HealthStatus:
        """Retrieve overall system health and component statuses.

        Returns:
            HealthStatus instance with startup, readiness, and liveness states.

        Raises:
            APIError: If the server returns an error.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "GET", "/health", fallback_path="/api/v1/health"
        )
        return HealthStatus.from_dict(data)

    def get_liveness(self) -> LiveStatus:
        """Query liveness probe endpoint.

        Returns:
            LiveStatus instance (supports dictionary-like access).

        Raises:
            APIError: If server returns an error.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "GET", "/health/live", fallback_path="/api/v1/health/liveness"
        )
        return LiveStatus.from_dict(data)

    def get_readiness(self) -> ReadyStatus:
        """Query readiness probe endpoint.

        Returns:
            ReadyStatus instance (supports dictionary-like access).

        Raises:
            APIError: If server returns an error.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "GET", "/health/ready", fallback_path="/api/v1/health/readiness"
        )
        return ReadyStatus.from_dict(data)

    def get_version(self) -> VersionInfo:
        """Retrieve API and core engine version metadata.

        Returns:
            VersionInfo instance containing version strings and git commit info.

        Raises:
            APIError: If server returns an error.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "GET", "/version", fallback_path="/api/v1/version"
        )
        return VersionInfo.from_dict(data)

    def submit_inference(
        self,
        video_path: str,
        enable_calibration: bool = False,
    ) -> JobStatus:
        """Submit a video for asynchronous crowd risk inference.

        Args:
            video_path: Path to the MP4 or AVI video file.
            enable_calibration: Whether to enable metric ground-plane calibration.

        Returns:
            JobStatus representing the queued or processing job.

        Raises:
            ValidationError: If input parameters or video file format are invalid.
            APIError: If submission fails on server.
            ConnectionError: If connection fails.
        """
        body, content_type = self._build_multipart_payload(
            video_path, enable_calibration
        )
        headers = {"Content-Type": content_type}
        data = self._request_with_fallback(
            "POST",
            "/inference",
            fallback_path="/api/v1/inference",
            body=body,
            headers=headers,
        )
        return JobStatus.from_dict(data)

    def get_job(self, job_id: str) -> JobStatus:
        """Query current execution status of an ongoing or completed job.

        Args:
            job_id: Unique job identifier string.

        Returns:
            JobStatus instance.

        Raises:
            NotFoundError: If job ID does not exist.
            APIError: If query fails.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "GET",
            f"/jobs/{job_id}",
            fallback_path=f"/api/v1/jobs/{job_id}",
        )
        # If response did not contain job_id, backfill it
        if "job_id" not in data:
            data = dict(data)
            data["job_id"] = job_id
        return JobStatus.from_dict(data)

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancel a running or queued inference job.

        Args:
            job_id: Identifier of the job to cancel.

        Returns:
            JobStatus reflecting the cancelled state.

        Raises:
            NotFoundError: If job ID does not exist.
            APIError: If cancellation fails.
            ConnectionError: If connection fails.
        """
        data = self._request_with_fallback(
            "DELETE",
            f"/jobs/{job_id}",
            fallback_path=f"/api/v1/jobs/{job_id}",
        )
        if not data or not isinstance(data, Mapping):
            return JobStatus(job_id=job_id, status="CANCELLED", progress=0.0)

        data_dict = dict(data)
        data_dict.setdefault("job_id", job_id)
        data_dict.setdefault("status", "CANCELLED")
        return JobStatus.from_dict(data_dict)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval_seconds: float = 0.5,
        timeout_seconds: float = 60.0,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> JobStatus:
        """Poll a job until completion, cancellation, or failure.

        Args:
            job_id: Identifier of the job to poll.
            poll_interval_seconds: Sleep interval between consecutive status checks.
            timeout_seconds: Maximum total wait duration before raising TimeoutError.
            poll_interval: Alias for poll_interval_seconds.
            timeout: Alias for timeout_seconds.

        Returns:
            JobStatus of the completed job.

        Raises:
            JobCancelledError: If the job was cancelled.
            APIError: If the job execution failed.
            TimeoutError: If the job did not complete within timeout_seconds.
        """
        if timeout is not None:
            timeout_seconds = timeout
        if poll_interval is not None:
            poll_interval_seconds = poll_interval

        deadline = time.monotonic() + timeout_seconds

        while True:
            job = self.get_job(job_id)

            if job.is_completed:
                return job

            if job.is_cancelled:
                raise JobCancelledError(
                    f"Job '{job_id}' was cancelled.",
                    job_id=job_id,
                )

            if job.is_failed:
                raise APIError(
                    message=f"Job '{job_id}' failed: {job.error or 'Unknown error'}",
                    status_code=500,
                    detail=job.error,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Job '{job_id}' timed out after {timeout_seconds:.1f} seconds."
                )

            time.sleep(min(poll_interval_seconds, remaining))
