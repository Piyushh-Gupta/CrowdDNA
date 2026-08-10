"""Unit tests for the VideoIngestor (crowdflow_dna/ingestion/video_loader.py).

Tests are written using pytest and unittest.mock to avoid requiring
real video files or heavy ML dependencies during CI runs.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna import config
from crowdflow_dna.errors import (
    InvalidVideoFormatError,
    UploadSizeExceededError,
    VideoCorruptionError,
)
from crowdflow_dna.ingestion.video_loader import VideoIngestor


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ingestor() -> VideoIngestor:
    """Return a fresh VideoIngestor instance."""
    return VideoIngestor()

@pytest.fixture()
def tmp_mp4(tmp_path: Path) -> Path:
    """Create a dummy .mp4 file."""
    f = tmp_path / "test_video.mp4"
    f.write_bytes(b"\x00" * 1024)
    return f

@pytest.fixture()
def tmp_avi(tmp_path: Path) -> Path:
    """Create a dummy .avi file."""
    f = tmp_path / "test_video.avi"
    f.write_bytes(b"\x00" * 1024)
    return f

def mock_ffprobe_result(
    fps: float = 25.0,
    width: int = 640,
    height: int = 480,
    frame_count: int = 250,
    duration: float = 10.0,
    returncode: int = 0
):
    """Return a mock subprocess.CompletedProcess for ffprobe."""
    stdout_dict = {
        "streams": [
            {
                "width": width,
                "height": height,
                "avg_frame_rate": f"{int(fps)}/1" if fps > 0 else "0/0",
                "nb_frames": str(frame_count) if frame_count else "N/A",
                "duration": str(duration)
            }
        ]
    }
    result = MagicMock()
    result.returncode = returncode
    result.stdout = json.dumps(stdout_dict)
    result.stderr = ""
    return result

def mock_ffmpeg_process(width=640, height=480, readable_frames=10, returncode=0, partial_last_frame=False):
    """Return a mock subprocess.Popen for ffmpeg."""
    process = MagicMock()
    process.returncode = returncode
    
    frame_bytes = width * height * 3
    dummy_frame = b"\x00" * frame_bytes
    
    chunks = [dummy_frame for _ in range(readable_frames)]
    if partial_last_frame:
        chunks.append(b"\x00" * (frame_bytes // 2))
        
    def read_generator():
        for chunk in chunks:
            # We mock process.stdout.read to return exactly what's requested if available.
            # A real file object read(n) returns up to n bytes.
            yield chunk
        while True:
            yield b""
            
    gen = read_generator()
    
    def side_effect(size=-1):
        try:
            return next(gen)
        except StopIteration:
            return b""

    process.stdout.read.side_effect = side_effect
    process.stderr.read.return_value = b"stderr dummy"
    process.poll.return_value = returncode
    return process


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_valid_mp4_extension_passes(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    mock_run.return_value = mock_ffprobe_result()
    mock_popen.return_value = mock_ffmpeg_process()
    
    import time
    frames_iter, _ = ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)
    frames = list(frames_iter)
    assert isinstance(frames, list)
    assert len(frames) > 0

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_valid_avi_extension_passes(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_avi: Path
) -> None:
    mock_run.return_value = mock_ffprobe_result()
    mock_popen.return_value = mock_ffmpeg_process()
    
    import time
    frames_iter, _ = ingestor.load(str(tmp_avi), deadline=time.monotonic() + 10)
    frames = list(frames_iter)
    assert isinstance(frames, list)
    assert len(frames) > 0

def test_invalid_extension_raises(ingestor: VideoIngestor, tmp_path: Path) -> None:
    bad_file = tmp_path / "video.mkv"
    bad_file.write_bytes(b"\x00" * 100)
    with pytest.raises(InvalidVideoFormatError, match="Unsupported file type"):
        import time
        ingestor.load(str(bad_file), deadline=time.monotonic() + 10)

def test_missing_file_raises(ingestor: VideoIngestor) -> None:
    with pytest.raises(VideoCorruptionError, match="File not found"):
        import time
        ingestor.load("/nonexistent/path/video.mp4", deadline=time.monotonic() + 10)

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_duration_exceeds_limit_raises(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_path: Path
) -> None:
    long_video = tmp_path / "long.mp4"
    long_video.write_bytes(b"\x00" * 1024)
    # 1 million frames at 25 fps = 40,000 seconds
    mock_run.return_value = mock_ffprobe_result(fps=25.0, frame_count=1_000_000, duration=40000.0)
    
    with pytest.raises(UploadSizeExceededError, match="duration"):
        import time
        ingestor.load(str(long_video), deadline=time.monotonic() + 10)

@patch("shutil.which", return_value=None)
def test_missing_ffprobe_raises(mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    with pytest.raises(VideoCorruptionError, match="ffprobe executable not found"):
        import time
        ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)

@patch("shutil.which", side_effect=lambda x: "/usr/bin/ffprobe" if x == "ffprobe" else None)
@patch("subprocess.run")
def test_missing_ffmpeg_raises(mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    mock_run.return_value = mock_ffprobe_result()
    with pytest.raises(VideoCorruptionError, match="ffmpeg executable not found"):
        import time
        list(ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)[0])

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
def test_ffprobe_timeout_raises(mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=15)
    with pytest.raises(VideoCorruptionError, match="ffprobe timed out"):
        import time
        ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
def test_ffprobe_malformed_json(mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    result = MagicMock()
    result.returncode = 0
    result.stdout = "NOT JSON"
    mock_run.return_value = result
    with pytest.raises(VideoCorruptionError, match="ffprobe output was not valid JSON"):
        import time
        ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_ffmpeg_non_zero_exit(mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    mock_run.return_value = mock_ffprobe_result()
    mock_popen.return_value = mock_ffmpeg_process(returncode=1)
    
    with pytest.raises(VideoCorruptionError, match="FFmpeg failed with exit code 1"):
        import time
        list(ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)[0])

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_ffmpeg_partial_frame_raises(mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path) -> None:
    mock_run.return_value = mock_ffprobe_result()
    mock_popen.return_value = mock_ffmpeg_process(partial_last_frame=True)
    
    with pytest.raises(VideoCorruptionError, match="Partial frame read"):
        import time
        list(ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)[0])


@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_metadata_keys_and_types(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    mock_run.return_value = mock_ffprobe_result(fps=30.0, width=1280, height=720, frame_count=300)
    mock_popen.return_value = mock_ffmpeg_process(width=1280, height=720, readable_frames=300)
    import time
    _, metadata = ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)

    assert set(metadata.keys()) == {
        "fps",
        "width",
        "height",
        "frame_count",
        "duration_seconds",
        "sample_rate",
    }
    assert isinstance(metadata["fps"], float)
    assert isinstance(metadata["width"], int)
    assert isinstance(metadata["height"], int)
    assert isinstance(metadata["frame_count"], int)
    assert isinstance(metadata["duration_seconds"], float)
    assert isinstance(metadata["sample_rate"], int)

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_frame_sampling_respects_sample_rate(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    readable_frames = 25
    mock_run.return_value = mock_ffprobe_result(frame_count=250)
    mock_popen.return_value = mock_ffmpeg_process(readable_frames=readable_frames)
    import time
    frames_iter, _ = ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)
    frames = list(frames_iter)

    expected_count = sum(
        1 for i in range(readable_frames) if i % config.FRAME_SAMPLE_RATE == 0
    )
    assert len(frames) == expected_count

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_frames_are_numpy_arrays(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    mock_run.return_value = mock_ffprobe_result(frame_count=100)
    mock_popen.return_value = mock_ffmpeg_process(readable_frames=10)
    import time
    frames_iter, _ = ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)
    frames = list(frames_iter)

    assert all(isinstance(f, np.ndarray) for f in frames)
    # Check shape: (height, width, 3)
    assert all(f.shape == (480, 640, 3) for f in frames)

@patch("shutil.which", return_value="/usr/bin/ffmpeg")
@patch("subprocess.run")
@patch("subprocess.Popen")
def test_pipes_closed_on_success(
    mock_popen, mock_run, mock_which, ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    mock_run.return_value = mock_ffprobe_result()
    process = mock_ffmpeg_process()
    mock_popen.return_value = process
    import time
    frames_iter, _ = ingestor.load(str(tmp_mp4), deadline=time.monotonic() + 10)
    list(frames_iter)
    
    process.stdout.close.assert_called_once()
    process.stderr.close.assert_called_once()
