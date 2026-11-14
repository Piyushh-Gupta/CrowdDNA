"""Comprehensive test suite for asynchronous inference execution and job management.

Covers:
- Pydantic schema validation and serialization
- JobManager lifecycle transitions (queued, running, completed, failed, cancelled)
- Job cancellation while queued and while running
- Safe temporary file cleanup on completion, cancellation, and error
- InferenceService bridging with CrowdFlowPipeline
- FastAPI route testing via TestClient (POST /inference, GET /jobs/{job_id}, DELETE /jobs/{job_id})
- Invalid file format validation
- Concurrent job execution and thread safety
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from crowdflow_dna.api.jobs.manager import (
    InvalidJobStateError,
    JobManager,
    reset_job_manager,
    set_job_manager,
)
from crowdflow_dna.api.routes.inference import router as inference_router
from crowdflow_dna.api.schemas.inference import (
    InferenceResultDTO,
    JobResponse,
    JobStatusEnum,
    RiskPredictionDTO,
    TimelineEntryDTO,
    VideoMetadataDTO,
)
from crowdflow_dna.api.services.inference_service import (
    InferenceService,
    set_inference_service,
)
from crowdflow_dna.errors import CrowdFlowError


# ---------------------------------------------------------------------------
# Test Fixtures & Dummy Objects
# ---------------------------------------------------------------------------


@dataclass
class DummyRiskPrediction:
    region_id: int
    label: str
    confidence: float


@dataclass
class DummyTimelineEntry:
    frame_index: int
    predictions: List[DummyRiskPrediction]


@dataclass
class DummyPipelineResult:
    output_video_path: str = "/tmp/dummy_output.mp4"
    timeline: List[DummyTimelineEntry] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.timeline is None:
            self.timeline = [
                DummyTimelineEntry(
                    frame_index=0,
                    predictions=[
                        DummyRiskPrediction(region_id=1, label="Safe", confidence=0.95)
                    ],
                ),
                DummyTimelineEntry(
                    frame_index=1,
                    predictions=[
                        DummyRiskPrediction(
                            region_id=1, label="Congesting", confidence=0.82
                        )
                    ],
                ),
            ]
        if self.metadata is None:
            self.metadata = {
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "frame_count": 60,
                "duration_seconds": 2.0,
                "sample_rate": 1,
            }


@pytest.fixture(autouse=True)
def clean_job_manager():
    """Ensure a fresh JobManager and InferenceService singleton for every test."""
    manager = JobManager(max_workers=4)
    set_job_manager(manager)
    service = InferenceService(job_manager=manager)
    set_inference_service(service)
    yield manager
    reset_job_manager()
    set_inference_service(None)


@pytest.fixture
def test_app():
    """Build a minimal FastAPI application with the inference router."""
    app = FastAPI(title="CrowdDNA API Test")
    app.include_router(inference_router)
    return app


@pytest.fixture
def client(test_app):
    """Provide a FastAPI TestClient instance."""
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# 1. Schema Tests
# ---------------------------------------------------------------------------


def test_job_status_enum_values():
    """Verify all 5 required job states exist."""
    assert JobStatusEnum.QUEUED.value == "queued"
    assert JobStatusEnum.RUNNING.value == "running"
    assert JobStatusEnum.COMPLETED.value == "completed"
    assert JobStatusEnum.FAILED.value == "failed"
    assert JobStatusEnum.CANCELLED.value == "cancelled"


def test_schema_serialization():
    """Test serialization and deserialization of API schemas."""
    risk_dto = RiskPredictionDTO(region_id=1, label="Critical", confidence=0.88)
    timeline_dto = TimelineEntryDTO(frame_index=5, predictions=[risk_dto])
    meta_dto = VideoMetadataDTO(
        fps=25.0,
        width=1280,
        height=720,
        frame_count=100,
        duration_seconds=4.0,
        sample_rate=1,
    )
    result_dto = InferenceResultDTO(
        output_video_path="/out/video.mp4",
        timeline=[timeline_dto],
        metadata=meta_dto,
    )
    job_resp = JobResponse(
        job_id="test-uuid",
        status=JobStatusEnum.COMPLETED,
        created_at=datetime.now(timezone.utc),
        result=result_dto,
    )

    data = job_resp.model_dump()
    assert data["job_id"] == "test-uuid"
    assert data["status"] == "completed"
    assert data["result"]["output_video_path"] == "/out/video.mp4"
    assert len(data["result"]["timeline"]) == 1
    assert data["result"]["timeline"][0]["predictions"][0]["label"] == "Critical"
    assert data["result"]["metadata"]["fps"] == 25.0


# ---------------------------------------------------------------------------
# 2. JobManager Lifecycle & Cancellation Tests
# ---------------------------------------------------------------------------


def test_job_manager_lifecycle_success(clean_job_manager: JobManager):
    """Test standard progression: queued -> running -> completed."""
    started_evt = threading.Event()
    finish_evt = threading.Event()

    def mock_task(job_id, video_path, enable_calibration, cancel_event):
        started_evt.set()
        finish_evt.wait(timeout=2.0)
        return InferenceResultDTO(output_video_path="/out/done.mp4")

    # Use a dummy temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"dummy video data")
        temp_path = tf.name

    resp = clean_job_manager.submit_job(video_path=temp_path, task_fn=mock_task)
    assert resp.status == JobStatusEnum.QUEUED
    assert resp.job_id is not None

    # Wait until running
    assert started_evt.wait(timeout=2.0)
    current = clean_job_manager.get_job(resp.job_id)
    assert current.status == JobStatusEnum.RUNNING
    assert current.started_at is not None

    finish_evt.set()
    time.sleep(0.1)

    completed = clean_job_manager.get_job(resp.job_id)
    assert completed.status == JobStatusEnum.COMPLETED
    assert completed.completed_at is not None
    assert completed.result.output_video_path == "/out/done.mp4"
    assert not Path(
        temp_path
    ).exists(), "Temporary file should be automatically cleaned up"


def test_job_manager_lifecycle_failure(clean_job_manager: JobManager):
    """Test failure progression: queued -> running -> failed."""
    started_evt = threading.Event()

    def failing_task(job_id, video_path, enable_calibration, cancel_event):
        started_evt.set()
        raise RuntimeError("Model inference crashed unexpectedly")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"data")
        temp_path = tf.name

    resp = clean_job_manager.submit_job(video_path=temp_path, task_fn=failing_task)
    assert started_evt.wait(timeout=2.0)
    time.sleep(0.1)

    failed = clean_job_manager.get_job(resp.job_id)
    assert failed.status == JobStatusEnum.FAILED
    assert "Model inference crashed unexpectedly" in failed.error
    assert failed.completed_at is not None
    assert not Path(temp_path).exists(), "Temp file should be deleted on failure"


def test_job_manager_cancellation_while_running(clean_job_manager: JobManager):
    """Test cancelling a currently running job."""
    running_evt = threading.Event()
    cancelled_observed = threading.Event()

    def long_task(job_id, video_path, enable_calibration, cancel_event):
        running_evt.set()
        # Wait until cancelled
        while not cancel_event.is_set():
            time.sleep(0.01)
        cancelled_observed.set()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tf.write(b"video")
        temp_path = tf.name

    resp = clean_job_manager.submit_job(video_path=temp_path, task_fn=long_task)
    assert running_evt.wait(timeout=2.0)

    # Cancel while running
    cancel_resp = clean_job_manager.cancel_job(resp.job_id)
    assert cancel_resp.status == JobStatusEnum.CANCELLED
    assert clean_job_manager.is_cancelled(resp.job_id)

    assert cancelled_observed.wait(timeout=2.0)
    time.sleep(0.1)

    final = clean_job_manager.get_job(resp.job_id)
    assert final.status == JobStatusEnum.CANCELLED
    assert not Path(temp_path).exists(), "Temp file should be deleted on cancel"


def test_job_manager_cancellation_while_queued():
    """Test cancelling a job before worker thread picks it up."""
    # Create single-worker manager to artificially keep second job in queue
    single_worker = JobManager(max_workers=1)
    blocker_started = threading.Event()
    unblock = threading.Event()

    def blocking_task(job_id, video_path, enable_calibration, cancel_event):
        blocker_started.set()
        unblock.wait(timeout=2.0)

    resp1 = single_worker.submit_job(video_path="v1.mp4", task_fn=blocking_task)
    assert resp1.job_id is not None
    assert blocker_started.wait(timeout=2.0)

    # Submit second job - will remain queued
    queued_task = MagicMock()
    resp2 = single_worker.submit_job(video_path="v2.mp4", task_fn=queued_task)
    assert single_worker.get_job(resp2.job_id).status == JobStatusEnum.QUEUED

    # Cancel queued job
    cancelled_resp = single_worker.cancel_job(resp2.job_id)
    assert cancelled_resp.status == JobStatusEnum.CANCELLED

    # Unblock first job and verify second never ran
    unblock.set()
    single_worker.shutdown(wait=True)
    queued_task.assert_not_called()


def test_job_manager_cancel_terminal_job(clean_job_manager: JobManager):
    """Test that cancelling already completed or failed job raises InvalidJobStateError."""
    done_evt = threading.Event()

    def quick_task(job_id, video_path, enable_calibration, cancel_event):
        done_evt.set()
        return InferenceResultDTO(output_video_path="out.mp4")

    resp = clean_job_manager.submit_job(video_path="video.mp4", task_fn=quick_task)
    assert done_evt.wait(timeout=2.0)
    time.sleep(0.1)

    assert clean_job_manager.get_job(resp.job_id).status == JobStatusEnum.COMPLETED

    with pytest.raises(InvalidJobStateError, match="Cannot cancel job"):
        clean_job_manager.cancel_job(resp.job_id)


def test_job_manager_cancel_nonexistent_job(clean_job_manager: JobManager):
    """Test cancelling non-existent job returns None."""
    assert clean_job_manager.cancel_job("non-existent-uuid") is None
    assert clean_job_manager.get_job("non-existent-uuid") is None


# ---------------------------------------------------------------------------
# 3. InferenceService Unit Tests
# ---------------------------------------------------------------------------


def test_inference_service_successful_run(clean_job_manager: JobManager):
    """Test InferenceService executes pipeline and converts result to DTO."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = DummyPipelineResult()

    service = InferenceService(
        job_manager=clean_job_manager,
        pipeline_factory=lambda **kw: mock_pipeline,
    )

    job_resp = clean_job_manager.submit_job(
        video_path="test_video.mp4",
        task_fn=service.run_inference,
    )

    # Wait for completion
    for _ in range(20):
        current = clean_job_manager.get_job(job_resp.job_id)
        if current.status == JobStatusEnum.COMPLETED:
            break
        time.sleep(0.05)

    assert current.status == JobStatusEnum.COMPLETED
    assert current.result is not None
    assert len(current.result.timeline) == 2
    assert current.result.metadata.width == 1920
    assert current.result.output_video_path == "/tmp/dummy_output.mp4"
    mock_pipeline.run.assert_called_once_with("test_video.mp4")


def test_inference_service_calibration_flags(clean_job_manager: JobManager):
    """Test InferenceService forwards calibration enabled/disabled correctly."""
    captured_configs = []

    def mock_pipeline_factory(**kwargs):
        captured_configs.append(kwargs.get("calibration_config"))
        mock = MagicMock()
        mock.run.return_value = DummyPipelineResult()
        return mock

    service = InferenceService(
        job_manager=clean_job_manager,
        pipeline_factory=mock_pipeline_factory,
    )

    # Run with enable_calibration = True
    job_true = clean_job_manager.submit_job(
        video_path="v_calib.mp4",
        enable_calibration=True,
        task_fn=service.run_inference,
    )
    for _ in range(20):
        if clean_job_manager.get_job(job_true.job_id).status == JobStatusEnum.COMPLETED:
            break
        time.sleep(0.05)

    assert captured_configs[0].enabled is True

    # Run with enable_calibration = False
    job_false = clean_job_manager.submit_job(
        video_path="v_no_calib.mp4",
        enable_calibration=False,
        task_fn=service.run_inference,
    )
    for _ in range(20):
        if (
            clean_job_manager.get_job(job_false.job_id).status
            == JobStatusEnum.COMPLETED
        ):
            break
        time.sleep(0.05)

    assert captured_configs[1].enabled is False


def test_inference_service_pipeline_failure(clean_job_manager: JobManager):
    """Test InferenceService marks job as failed when pipeline raises CrowdFlowError."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = CrowdFlowError("Corrupt frame decoding")

    service = InferenceService(
        job_manager=clean_job_manager,
        pipeline_factory=lambda **kw: mock_pipeline,
    )

    job_resp = clean_job_manager.submit_job(
        video_path="corrupt.mp4",
        task_fn=service.run_inference,
    )

    for _ in range(20):
        current = clean_job_manager.get_job(job_resp.job_id)
        if current.status == JobStatusEnum.FAILED:
            break
        time.sleep(0.05)

    assert current.status == JobStatusEnum.FAILED
    assert "Corrupt frame decoding" in current.error


# ---------------------------------------------------------------------------
# 4. FastAPI Route Integration Tests (HTTP Client)
# ---------------------------------------------------------------------------


def test_post_inference_valid_mp4(client: TestClient):
    """POST /inference accepts .mp4 upload and returns 202 with status 'queued'."""
    video_bytes = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
    files = {"file": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")}

    response = client.post("/inference", files=files)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    assert data["created_at"] is not None


def test_post_inference_valid_avi(client: TestClient):
    """POST /inference accepts .avi upload and returns 202."""
    avi_bytes = b"RIFF....AVI " + b"\x00" * 50
    files = {"file": ("test.avi", io.BytesIO(avi_bytes), "video/x-msvideo")}

    response = client.post("/inference", files=files)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data


def test_post_inference_invalid_extension(client: TestClient):
    """POST /inference rejects non-video files (.txt, .mkv) with 400 Bad Request."""
    files = {"file": ("malicious.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/inference", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_post_inference_with_calibration_flag(client: TestClient):
    """POST /inference accepts enable_calibration flag via Form."""
    video_bytes = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
    files = {"file": ("calib_video.mp4", io.BytesIO(video_bytes), "video/mp4")}
    data = {"enable_calibration": "true"}

    response = client.post("/inference", files=files, data=data)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"


def test_get_job_polling_completion(client: TestClient, clean_job_manager: JobManager):
    """Submit job and poll GET /jobs/{job_id} until completed."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = DummyPipelineResult()

    test_service = InferenceService(
        job_manager=clean_job_manager,
        pipeline_factory=lambda **kw: mock_pipeline,
    )
    set_inference_service(test_service)

    video_bytes = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
    files = {"file": ("poll_video.mp4", io.BytesIO(video_bytes), "video/mp4")}

    create_resp = client.post("/inference", files=files)
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]

    # Poll status
    completed_data = None
    for _ in range(30):
        get_resp = client.get(f"/jobs/{job_id}")
        assert get_resp.status_code == 200
        completed_data = get_resp.json()
        if completed_data["status"] == "completed":
            break
        time.sleep(0.05)

    assert completed_data["status"] == "completed"
    assert completed_data["result"] is not None
    assert completed_data["result"]["output_video_path"] == "/tmp/dummy_output.mp4"
    assert len(completed_data["result"]["timeline"]) == 2
    assert completed_data["result"]["metadata"]["fps"] == 30.0


def test_get_job_not_found(client: TestClient):
    """GET /jobs/{job_id} returns 404 for unknown job ID."""
    response = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_job_cancellation(client: TestClient, clean_job_manager: JobManager):
    """DELETE /jobs/{job_id} cancels a running job and returns status 'cancelled'."""
    blocker = threading.Event()

    def slow_task(job_id, video_path, enable_calibration, cancel_event):
        blocker.wait(timeout=2.0)

    # Submit job using slow task
    resp = clean_job_manager.submit_job(video_path="slow.mp4", task_fn=slow_task)
    job_id = resp.job_id

    # Call DELETE /jobs/{job_id}
    del_resp = client.delete(f"/jobs/{job_id}")
    assert del_resp.status_code == 200
    data = del_resp.json()
    assert data["status"] == "cancelled"

    # Polling GET should also show cancelled
    get_resp = client.get(f"/jobs/{job_id}")
    assert get_resp.json()["status"] == "cancelled"

    blocker.set()


def test_delete_job_not_found(client: TestClient):
    """DELETE /jobs/{job_id} returns 404 for unknown job ID."""
    response = client.delete("/jobs/unknown-job-id")
    assert response.status_code == 404


def test_delete_job_already_completed(
    client: TestClient, clean_job_manager: JobManager
):
    """DELETE /jobs/{job_id} on completed job returns 400 Bad Request."""

    def instant_task(job_id, video_path, enable_calibration, cancel_event):
        return InferenceResultDTO(output_video_path="done.mp4")

    resp = clean_job_manager.submit_job(video_path="instant.mp4", task_fn=instant_task)
    time.sleep(0.1)

    assert clean_job_manager.get_job(resp.job_id).status == JobStatusEnum.COMPLETED

    del_resp = client.delete(f"/jobs/{resp.job_id}")
    assert del_resp.status_code == 400
    assert "Cannot cancel job" in del_resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Concurrent Requests & Thread Safety
# ---------------------------------------------------------------------------


def test_concurrent_job_execution(client: TestClient, clean_job_manager: JobManager):
    """Verify concurrent submissions, non-blocking execution, and proper cleanup."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = DummyPipelineResult()

    test_service = InferenceService(
        job_manager=clean_job_manager,
        pipeline_factory=lambda **kw: mock_pipeline,
    )
    set_inference_service(test_service)

    num_jobs = 6
    job_ids = []

    def submit_one(i: int) -> str:
        data = io.BytesIO(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 50)
        files = {"file": (f"concurrent_{i}.mp4", data, "video/mp4")}
        res = client.post("/inference", files=files)
        assert res.status_code == 202
        return res.json()["job_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as submitter:
        futures = [submitter.submit(submit_one, i) for i in range(num_jobs)]
        job_ids = [f.result() for f in futures]

    assert len(set(job_ids)) == num_jobs, "All submitted jobs must have distinct UUIDs"

    # Poll until all jobs reach COMPLETED state
    deadline = time.time() + 5.0
    completed_ids = set()
    while time.time() < deadline and len(completed_ids) < num_jobs:
        for jid in job_ids:
            if jid in completed_ids:
                continue
            r = client.get(f"/jobs/{jid}")
            if r.status_code == 200 and r.json()["status"] == "completed":
                completed_ids.add(jid)
        time.sleep(0.05)

    assert (
        len(completed_ids) == num_jobs
    ), f"Only {len(completed_ids)}/{num_jobs} completed in time"

    # Verify no temp upload files remain on disk for these jobs
    for jid in job_ids:
        record = clean_job_manager._jobs.get(jid)
        if record:
            for p in record.temp_files:
                assert not p.exists()


def test_saturated_queue_and_concurrent_cancellation():
    """Verify behavior when worker pool is saturated and queued jobs are cancelled."""
    manager = JobManager(max_workers=2)
    started_jobs = []
    started_lock = threading.Lock()
    unblock = threading.Event()

    def slow_task(job_id, video_path, enable_calibration, cancel_event):
        with started_lock:
            started_jobs.append(job_id)
        unblock.wait(timeout=2.0)
        return InferenceResultDTO(output_video_path="done.mp4")

    # Create 5 temp files and submit 5 jobs to a 2-worker pool
    temp_files = []
    job_ids = []
    for i in range(5):
        tf = tempfile.NamedTemporaryFile(suffix=f"_sat_{i}.mp4", delete=False)
        tf.write(b"data")
        tf.close()
        temp_files.append(Path(tf.name))
        resp = manager.submit_job(
            video_path=tf.name,
            task_fn=slow_task,
            temp_files=[Path(tf.name)],
        )
        job_ids.append(resp.job_id)

    # Wait until 2 workers start executing
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with started_lock:
            if len(started_jobs) == 2:
                break
        time.sleep(0.02)

    # Cancel jobs 2 and 3 (queued) and job 0 (running)
    manager.cancel_job(job_ids[2])
    manager.cancel_job(job_ids[3])
    manager.cancel_job(job_ids[0])

    assert manager.get_job(job_ids[2]).status == JobStatusEnum.CANCELLED
    assert manager.get_job(job_ids[3]).status == JobStatusEnum.CANCELLED
    assert manager.get_job(job_ids[0]).status == JobStatusEnum.CANCELLED

    # Unblock the running workers and shut down
    unblock.set()
    manager.shutdown(wait=True)

    # Verify all 5 temporary files are cleanly deleted from disk
    for tf_path in temp_files:
        assert not tf_path.exists(), f"File {tf_path} was not cleaned up!"


# ---------------------------------------------------------------------------
# 5. Video Streaming & Upload Limit Tests
# ---------------------------------------------------------------------------


def test_get_job_video_successful_streaming(
    client: TestClient,
    clean_job_manager: JobManager,
    tmp_path: Path,
):
    """Test GET /jobs/{job_id}/video returns 200 with FileResponse for completed jobs."""
    # Create fake completed video file
    out_video = tmp_path / "rendered_output.mp4"
    out_video.write_bytes(b"\x00\x00\x00\x1cftypisom" + b"A" * 100)

    # Submit job and mark completed
    resp = clean_job_manager.submit_job(
        video_path="input.mp4",
        task_fn=lambda **kw: InferenceResultDTO(output_video_path=str(out_video)),
    )
    time.sleep(0.1)

    r = client.get(f"/jobs/{resp.job_id}/video")
    assert r.status_code == 200
    assert "video/mp4" in r.headers.get("content-type", "")
    assert r.headers.get("accept-ranges") == "bytes"
    assert len(r.content) == len(out_video.read_bytes())


def test_get_job_video_not_found(client: TestClient):
    """Test GET /jobs/{job_id}/video returns 404 for unknown job."""
    r = client.get("/jobs/00000000-0000-0000-0000-000000000000/video")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_get_job_video_not_completed(
    client: TestClient,
    clean_job_manager: JobManager,
):
    """Test GET /jobs/{job_id}/video returns 400 when job is not yet completed."""
    hold_event = threading.Event()

    def blocked_task(job_id, video_path, enable_calibration, cancel_event):
        hold_event.wait(timeout=2.0)
        return InferenceResultDTO(output_video_path="out.mp4")

    resp = clean_job_manager.submit_job(
        video_path="video.mp4",
        task_fn=blocked_task,
    )

    try:
        r = client.get(f"/jobs/{resp.job_id}/video")
        assert r.status_code == 400
        assert "only available for completed jobs" in r.json()["detail"].lower()
    finally:
        hold_event.set()


def test_get_job_video_missing_on_disk(
    client: TestClient,
    clean_job_manager: JobManager,
):
    """Test GET /jobs/{job_id}/video returns 404 when file was deleted from disk."""
    resp = clean_job_manager.submit_job(
        video_path="input.mp4",
        task_fn=lambda **kw: InferenceResultDTO(output_video_path="/nonexistent/missing.mp4"),
    )
    time.sleep(0.1)

    r = client.get(f"/jobs/{resp.job_id}/video")
    assert r.status_code == 404
    assert "not found on server storage" in r.json()["detail"].lower()


def test_submit_inference_upload_size_limit_rejection(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that uploads exceeding MAX_UPLOAD_SIZE_BYTES are rejected immediately with 413."""
    import crowdflow_dna.api.routes.inference as inf_route

    # Set artificial small limit of 500 bytes
    monkeypatch.setattr(inf_route, "MAX_UPLOAD_SIZE_BYTES", 500)

    # 1. Above limit (800 bytes) -> 413
    oversized_data = b"X" * 800
    r_over = client.post(
        "/inference",
        files={"file": ("large.mp4", oversized_data, "video/mp4")},
    )
    assert r_over.status_code == 413
    assert "exceeds" in r_over.json()["detail"].lower()

    # 2. Exactly at limit (500 bytes) -> 202 Accepted
    exact_data = b"Y" * 500
    r_exact = client.post(
        "/inference",
        files={"file": ("exact.mp4", exact_data, "video/mp4")},
    )
    assert r_exact.status_code == 202

    # 3. Below limit (200 bytes) -> 202 Accepted
    below_data = b"Z" * 200
    r_below = client.post(
        "/inference",
        files={"file": ("below.mp4", below_data, "video/mp4")},
    )
    assert r_below.status_code == 202
