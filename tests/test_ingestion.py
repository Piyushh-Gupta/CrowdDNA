"""Unit tests for the VideoIngestor (crowdflow_dna/ingestion/video_loader.py).

Tests are written using pytest and unittest.mock to avoid requiring
real video files or heavy ML dependencies during CI runs.
"""

import sys
from pathlib import Path
from typing import Any, Dict
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


def _make_mock_cap(
    fps: float = 25.0,
    width: int = 640,
    height: int = 480,
    frame_count: int = 250,
    readable_frames: int = 10,
    is_opened: bool = True,
) -> MagicMock:
    """Build a MagicMock that imitates cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = is_opened

    import cv2  # local import to avoid hard dependency at module level

    prop_map = {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: float(width),
        cv2.CAP_PROP_FRAME_HEIGHT: float(height),
        cv2.CAP_PROP_FRAME_COUNT: float(frame_count),
    }
    cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    read_returns = [(True, dummy_frame)] * readable_frames + [(False, None)]
    cap.read.side_effect = read_returns

    return cap


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


# ---------------------------------------------------------------------------
# Extension validation tests
# ---------------------------------------------------------------------------


def test_valid_mp4_extension_passes(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """A .mp4 file should not raise an extension error."""
    mock_cap = _make_mock_cap()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(tmp_mp4))
    assert isinstance(frames, list)


def test_valid_avi_extension_passes(
    ingestor: VideoIngestor, tmp_avi: Path
) -> None:
    """A .avi file should not raise an extension error."""
    mock_cap = _make_mock_cap()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(tmp_avi))
    assert isinstance(frames, list)


def test_invalid_extension_raises(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """An unsupported extension must raise InvalidVideoFormatError."""
    bad_file = tmp_path / "video.mkv"
    bad_file.write_bytes(b"\x00" * 100)
    with pytest.raises(InvalidVideoFormatError, match="Unsupported file type"):
        ingestor.load(str(bad_file))


def test_txt_extension_raises(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """A .txt file must raise InvalidVideoFormatError."""
    txt_file = tmp_path / "not_a_video.txt"
    txt_file.write_text("hello")
    with pytest.raises(InvalidVideoFormatError):
        ingestor.load(str(txt_file))


# ---------------------------------------------------------------------------
# File size validation tests
# ---------------------------------------------------------------------------


def test_oversized_file_raises(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """A file exceeding MAX_FILE_SIZE_MB must raise UploadSizeExceededError."""
    big_file = tmp_path / "big.mp4"
    oversized_bytes = (config.MAX_FILE_SIZE_MB * 1024 * 1024) + 1
    big_file.write_bytes(b"\x00" * oversized_bytes)
    with pytest.raises(UploadSizeExceededError, match="exceeds the maximum"):
        ingestor.load(str(big_file))


def test_file_at_size_limit_passes(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """A file exactly at the size limit should be accepted."""
    exact_file = tmp_path / "exact.mp4"
    exact_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    exact_file.write_bytes(b"\x00" * exact_bytes)
    mock_cap = _make_mock_cap()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(exact_file))
    assert isinstance(frames, list)


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------


def test_missing_file_raises(ingestor: VideoIngestor) -> None:
    """A path pointing to a non-existent file must raise VideoCorruptionError."""
    with pytest.raises(VideoCorruptionError, match="File not found"):
        ingestor.load("/nonexistent/path/video.mp4")


# ---------------------------------------------------------------------------
# VideoCapture open failure
# ---------------------------------------------------------------------------


def test_opencv_open_failure_raises(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """If OpenCV cannot open the video, VideoCorruptionError must be raised."""
    mock_cap = _make_mock_cap(is_opened=False)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(VideoCorruptionError, match="could not open"):
            ingestor.load(str(tmp_mp4))


def test_cap_released_on_open_failure(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """VideoCapture.release() must be called even when isOpened() is False."""
    mock_cap = _make_mock_cap(is_opened=False)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(VideoCorruptionError):
            ingestor.load(str(tmp_mp4))
    mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Metadata validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"fps": 0.0}, "fps=0"),
        ({"width": 0}, "width=0"),
        ({"height": 0}, "height=0"),
        ({"frame_count": 0}, "frame_count=0"),
    ],
)
def test_invalid_stream_metadata_raises(
    ingestor: VideoIngestor,
    tmp_mp4: Path,
    bad_kwargs: Dict[str, Any],
    match: str,
) -> None:
    """Zero or negative stream metadata values must raise VideoCorruptionError."""
    mock_cap = _make_mock_cap(**bad_kwargs)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(VideoCorruptionError, match=match):
            ingestor.load(str(tmp_mp4))


# ---------------------------------------------------------------------------
# Duration validation tests
# ---------------------------------------------------------------------------


def test_duration_exceeds_limit_raises(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """A video longer than MAX_DURATION_SECONDS must raise UploadSizeExceededError."""
    long_video = tmp_path / "long.mp4"
    long_video.write_bytes(b"\x00" * 1024)
    mock_cap = _make_mock_cap(fps=25.0, frame_count=1_000_000)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(UploadSizeExceededError, match="duration"):
            ingestor.load(str(long_video))


def test_duration_at_limit_passes(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """A video whose duration equals MAX_DURATION_SECONDS should be accepted."""
    fps = 25.0
    frame_count = int(config.MAX_DURATION_SECONDS * fps)
    mock_cap = _make_mock_cap(fps=fps, frame_count=frame_count, readable_frames=5)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, metadata = ingestor.load(str(tmp_mp4))
    assert metadata["duration_seconds"] == pytest.approx(
        config.MAX_DURATION_SECONDS, rel=1e-3
    )


# ---------------------------------------------------------------------------
# Metadata dictionary contract tests
# ---------------------------------------------------------------------------


def test_metadata_keys_and_types(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """Returned metadata must have exactly the contracted keys with correct types."""
    mock_cap = _make_mock_cap(fps=30.0, width=1280, height=720, frame_count=300)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        _, metadata = ingestor.load(str(tmp_mp4))

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


def test_metadata_values_correct(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """Metadata values must match the simulated stream properties."""
    fps, width, height, frame_count = 24.0, 1920, 1080, 240
    mock_cap = _make_mock_cap(
        fps=fps, width=width, height=height, frame_count=frame_count
    )
    with patch("cv2.VideoCapture", return_value=mock_cap):
        _, metadata = ingestor.load(str(tmp_mp4))

    assert metadata["fps"] == pytest.approx(fps)
    assert metadata["width"] == width
    assert metadata["height"] == height
    assert metadata["frame_count"] == frame_count
    assert metadata["duration_seconds"] == pytest.approx(frame_count / fps)
    assert metadata["sample_rate"] == config.FRAME_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Frame sampling tests
# ---------------------------------------------------------------------------


def test_frame_sampling_respects_sample_rate(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """Number of returned frames must respect FRAME_SAMPLE_RATE."""
    readable_frames = 25
    mock_cap = _make_mock_cap(readable_frames=readable_frames, frame_count=250)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(tmp_mp4))

    expected_count = sum(
        1 for i in range(readable_frames) if i % config.FRAME_SAMPLE_RATE == 0
    )
    assert len(frames) == expected_count


def test_frames_are_numpy_arrays(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """Every returned frame must be a NumPy ndarray."""
    mock_cap = _make_mock_cap(readable_frames=10, frame_count=100)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(tmp_mp4))

    assert all(isinstance(f, np.ndarray) for f in frames)


def test_no_opencv_objects_returned(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """The returned tuple must NOT contain any cv2.VideoCapture objects."""
    import cv2  # local import

    mock_cap = _make_mock_cap()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, metadata = ingestor.load(str(tmp_mp4))

    assert not isinstance(frames, cv2.VideoCapture)
    assert not isinstance(metadata, cv2.VideoCapture)


def test_empty_video_returns_empty_frames(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """A video with no readable frames should return an empty list."""
    mock_cap = _make_mock_cap(readable_frames=0, frame_count=10)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames, _ = ingestor.load(str(tmp_mp4))
    assert frames == []


# ---------------------------------------------------------------------------
# Resource management: cap.release() always called
# ---------------------------------------------------------------------------


def test_cap_released_on_success(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """VideoCapture.release() must be called after a successful load."""
    mock_cap = _make_mock_cap()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        ingestor.load(str(tmp_mp4))
    mock_cap.release.assert_called_once()


def test_cap_released_on_bad_metadata(
    ingestor: VideoIngestor, tmp_mp4: Path
) -> None:
    """VideoCapture.release() must be called even when metadata validation fails."""
    mock_cap = _make_mock_cap(fps=0.0)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(VideoCorruptionError):
            ingestor.load(str(tmp_mp4))
    mock_cap.release.assert_called_once()


def test_cap_released_on_duration_exceeded(
    ingestor: VideoIngestor, tmp_path: Path
) -> None:
    """VideoCapture.release() must be called even when duration validation fails."""
    long_video = tmp_path / "long.mp4"
    long_video.write_bytes(b"\x00" * 1024)
    mock_cap = _make_mock_cap(fps=25.0, frame_count=1_000_000)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(UploadSizeExceededError):
            ingestor.load(str(long_video))
    mock_cap.release.assert_called_once()
