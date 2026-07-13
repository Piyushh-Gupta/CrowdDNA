"""Video ingestion module for CrowdFlow DNA.

This module validates uploaded video files, extracts metadata, and
samples frames for downstream processing in the detection module.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from crowdflow_dna import config
from crowdflow_dna.errors import (
    InvalidVideoFormatError,
    UploadSizeExceededError,
    VideoCorruptionError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".mp4", ".avi"})


class VideoIngestor:
    """Ingests, validates, and samples video files.

    Extracts frames and stream-level metadata for downstream processing.
    """

    def __init__(self) -> None:
        self.max_file_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        self.max_duration_seconds = float(config.MAX_DURATION_SECONDS)
        self.frame_sample_rate = config.FRAME_SAMPLE_RATE

    def load(self, video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Load, validate, and sample a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            A tuple of (frames, metadata).
                - frames: List of sampled np.ndarray frames (BGR).
                - metadata: Dictionary containing stream metadata.

        Raises:
            InvalidVideoFormatError: If the extension is unsupported.
            UploadSizeExceededError: If the file size or duration is too large.
            VideoCorruptionError: If the video cannot be opened or metadata is invalid.
        """
        path = Path(video_path)
        logger.info("Ingesting video: %s", path)

        self._validate_extension(path)
        self._validate_file_size(path)

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise VideoCorruptionError(
                    f"OpenCV could not open video: {path}. "
                    "The file may be corrupted or unreadable."
                )

            metadata = self._extract_metadata(cap, path)
            self._validate_duration(metadata["duration_seconds"], path)
            frames = self._sample_frames(cap, metadata)
        finally:
            cap.release()
            logger.debug("VideoCapture released for: %s", path)

        logger.info(
            "Ingestion complete: %d frames sampled from %s (%.1fs @ %.1f fps)",
            len(frames),
            path.name,
            metadata["duration_seconds"],
            metadata["fps"],
        )
        return frames, metadata

    def _validate_extension(self, path: Path) -> None:
        """Validate the file extension."""
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidVideoFormatError(
                f"Unsupported file type '{ext}'. "
                f"Please upload an MP4 or AVI video."
            )

    def _validate_file_size(self, path: Path) -> None:
        """Validate the file size against the configured maximum."""
        if not path.exists():
            raise VideoCorruptionError(f"File not found: {path}")

        size_bytes = os.path.getsize(path)
        if size_bytes > self.max_file_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            raise UploadSizeExceededError(
                f"File size {size_mb:.1f} MB exceeds the maximum allowed "
                f"{config.MAX_FILE_SIZE_MB} MB."
            )

    def _validate_duration(self, duration_seconds: float, path: Path) -> None:
        """Validate the video duration against the configured maximum."""
        if duration_seconds > self.max_duration_seconds:
            raise UploadSizeExceededError(
                f"Video duration {duration_seconds:.1f}s exceeds the maximum "
                f"allowed {config.MAX_DURATION_SECONDS}s."
            )

    def _extract_metadata(
        self, cap: cv2.VideoCapture, path: Path
    ) -> Dict[str, Any]:
        """Extract and validate stream metadata."""
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        issues: List[str] = []
        if fps <= 0:
            issues.append(f"fps={fps}")
        if width <= 0:
            issues.append(f"width={width}")
        if height <= 0:
            issues.append(f"height={height}")
        if frame_count <= 0:
            issues.append(f"frame_count={frame_count}")

        if issues:
            raise VideoCorruptionError(
                f"Invalid stream metadata for '{path}': " + ", ".join(issues)
            )

        duration_seconds = frame_count / fps

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_seconds": duration_seconds,
            "sample_rate": self.frame_sample_rate,
        }

    def _sample_frames(
        self, cap: cv2.VideoCapture, metadata: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Iterate through the video and sample frames based on config."""
        frames: List[np.ndarray] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self.frame_sample_rate == 0:
                frames.append(frame)

            frame_index += 1

        return frames
