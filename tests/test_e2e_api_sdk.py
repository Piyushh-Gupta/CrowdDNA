"""End-to-end integration tests connecting CrowdDNAClient SDK to FastAPI server.

Tests full HTTP lifecycle over real local socket connections:
- Health and readiness probes
- Asynchronous video inference submission and polling
- Cancellation workflows
- Error handling and HTTP mapping
- Temporary artifact cleanup
"""

from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import uvicorn

# Ensure repository root is on sys.path for test execution
REPO_ROOT = str(Path(__file__).parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crowdflow_dna.api.app import create_app  # noqa: E402
from crowdflow_dna.api.jobs.manager import JobManager, set_job_manager  # noqa: E402
from crowdflow_dna.api.schemas.inference import (  # noqa: E402
    InferenceResultDTO,
    RiskPredictionDTO,
    TimelineEntryDTO,
    VideoMetadataDTO,
)
from crowdflow_dna.api.services.inference_service import (  # noqa: E402
    InferenceService,
    set_inference_service,
)
from crowdflow_dna_sdk import (  # noqa: E402
    CrowdDNAClient,
    JobCancelledError,
    NotFoundError,
    ValidationError,
)


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_server() -> Generator[str, None, None]:
    """Spin up a live Uvicorn ASGI server running create_app() in background thread."""
    port = _find_free_port()
    app = create_app()

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait until the server accepts TCP connections
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5.0
    started = False
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                started = True
                break
        except OSError:
            time.sleep(0.05)

    if not started:
        server.should_exit = True
        raise RuntimeError("Failed to start live Uvicorn server within 5 seconds.")

    try:
        yield base_url
    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)


@pytest.fixture
def fresh_job_manager() -> Generator[JobManager, None, None]:
    """Provide a fresh JobManager instance for isolated testing."""
    manager = JobManager(max_workers=2)
    set_job_manager(manager)
    try:
        yield manager
    finally:
        manager.shutdown(wait=False)
        set_job_manager(None)
        set_inference_service(None)


@pytest.fixture
def sample_mp4_file(tmp_path: Path) -> Path:
    """Create a temporary dummy .mp4 file."""
    video_file = tmp_path / "test_sample.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41")
    return video_file


def test_e2e_system_endpoints(live_server: str) -> None:
    """Verify SDK methods against system health and version endpoints."""
    client = CrowdDNAClient(base_url=live_server)

    # 1. /health
    health = client.get_health()
    assert health.status == "healthy"
    assert health.version == "1.0.0"

    # 2. /health/live
    live = client.get_liveness()
    assert live.alive is True

    # 3. /health/ready
    ready = client.get_readiness()
    assert ready.ready is True
    assert ready.model_loaded is False

    # 4. /version
    version = client.get_version()
    assert version.api_version == "1.0.0"
    assert version.core_version == "1.0.0"


def test_e2e_inference_submission_and_wait(
    live_server: str,
    fresh_job_manager: JobManager,
    sample_mp4_file: Path,
) -> None:
    """Verify asynchronous video inference submission, polling, and results."""
    # Mock inference service returning completed result
    mock_service = MagicMock(spec=InferenceService)

    def dummy_run_inference(
        job_id: str,
        video_path: str,
        enable_calibration: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> InferenceResultDTO:
        time.sleep(0.1)  # Simulate brief processing
        return InferenceResultDTO(
            output_video_path=video_path,
            timeline=[
                TimelineEntryDTO(
                    frame_index=0,
                    predictions=[
                        RiskPredictionDTO(
                            region_id=1,
                            label="Safe",
                            confidence=0.95,
                        )
                    ],
                )
            ],
            metadata=VideoMetadataDTO(
                fps=30.0,
                width=1920,
                height=1080,
                frame_count=100,
                duration_seconds=3.33,
            ),
        )

    mock_service.run_inference.side_effect = dummy_run_inference
    set_inference_service(mock_service)

    client = CrowdDNAClient(base_url=live_server)

    # Submit inference job
    job = client.submit_inference(
        video_path=str(sample_mp4_file),
        enable_calibration=True,
    )
    assert job.job_id is not None
    assert job.is_done is False
    assert job.status in ("queued", "running", "PENDING", "PROCESSING")

    # Poll and wait for completion
    completed_job = client.wait_for_job(
        job_id=job.job_id,
        timeout=5.0,
        poll_interval=0.05,
    )
    assert completed_job.is_done is True
    assert completed_job.status in ("completed", "COMPLETED")
    assert completed_job.result is not None
    assert len(completed_job.result.timeline) == 1
    assert completed_job.result.timeline[0].predictions[0].label == "Safe"
    assert completed_job.result.metadata is not None
    assert completed_job.result.metadata.fps == 30.0


def test_e2e_inference_cancellation(
    live_server: str,
    fresh_job_manager: JobManager,
    sample_mp4_file: Path,
) -> None:
    """Verify cancellation of an in-progress or queued inference job."""
    started_event = threading.Event()
    cancel_acknowledged = threading.Event()

    mock_service = MagicMock(spec=InferenceService)

    def blocking_inference(
        job_id: str,
        video_path: str,
        enable_calibration: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        started_event.set()
        # Wait until cancelled or timeout
        if cancel_event:
            cancelled = cancel_event.wait(timeout=3.0)
            if cancelled:
                cancel_acknowledged.set()
        return None

    mock_service.run_inference.side_effect = blocking_inference
    set_inference_service(mock_service)

    client = CrowdDNAClient(base_url=live_server)

    # Submit job
    job = client.submit_inference(video_path=str(sample_mp4_file))
    assert job.job_id is not None

    # Wait until background execution starts
    assert started_event.wait(timeout=3.0) is True

    # Cancel the running job
    cancelled_job = client.cancel_job(job.job_id)
    assert cancelled_job.status in ("cancelled", "CANCELLED")

    # Confirm wait_for_job raises JobCancelledError
    with pytest.raises(JobCancelledError):
        client.wait_for_job(job.job_id, timeout=3.0, poll_interval=0.05)


def test_e2e_temporary_file_cleanup(
    live_server: str,
    fresh_job_manager: JobManager,
    sample_mp4_file: Path,
) -> None:
    """Verify uploaded temporary files are deleted after job processing."""
    uploaded_temp_file: list[str] = []

    mock_service = MagicMock(spec=InferenceService)

    def capture_temp_file(
        job_id: str,
        video_path: str,
        enable_calibration: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> InferenceResultDTO:
        uploaded_temp_file.append(video_path)
        # Temp file must exist while job is running
        assert os.path.exists(video_path)
        return InferenceResultDTO(
            output_video_path=video_path,
            timeline=[],
            metadata=VideoMetadataDTO(),
        )

    mock_service.run_inference.side_effect = capture_temp_file
    set_inference_service(mock_service)

    client = CrowdDNAClient(base_url=live_server)

    job = client.submit_inference(video_path=str(sample_mp4_file))
    client.wait_for_job(job.job_id, timeout=5.0, poll_interval=0.05)

    assert len(uploaded_temp_file) == 1
    temp_path = uploaded_temp_file[0]
    # Allow background cleaner a moment to unlink
    deadline = time.time() + 2.0
    while os.path.exists(temp_path) and time.time() < deadline:
        time.sleep(0.05)

    # Uploaded temporary file must be cleaned up
    assert not os.path.exists(temp_path)


def test_e2e_validation_error_on_unsupported_file(
    live_server: str,
    tmp_path: Path,
) -> None:
    """Verify submitting an unsupported file format raises ValidationError."""
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("not a video")

    client = CrowdDNAClient(base_url=live_server)

    with pytest.raises(ValidationError) as exc_info:
        client.submit_inference(video_path=str(unsupported))

    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in str(exc_info.value)


def test_e2e_not_found_error_on_unknown_job(live_server: str) -> None:
    """Verify querying an unknown job ID raises NotFoundError."""
    client = CrowdDNAClient(base_url=live_server)

    with pytest.raises(NotFoundError) as exc_info:
        client.get_job("00000000-0000-0000-0000-000000000000")

    assert exc_info.value.status_code == 404
