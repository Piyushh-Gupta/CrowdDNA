"""Video ingestion module for CrowdFlow DNA.

This module validates uploaded video files, extracts metadata using FFprobe,
and samples frames using an FFmpeg subprocess for downstream processing.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Optional

import numpy as np

from crowdflow_dna import config
from crowdflow_dna.errors import (
    InvalidVideoFormatError,
    UploadSizeExceededError,
    VideoCorruptionError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".mp4", ".avi"})


def get_rss_mb() -> Optional[float]:
    """Return process RSS in MB on Linux, or None if unavailable."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return None


class VideoIngestor:
    """Ingests, validates, and samples video files.

    Extracts frames and stream-level metadata using FFmpeg subprocesses.
    """

    def __init__(self) -> None:
        self.max_file_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        self.max_duration_seconds = float(config.MAX_DURATION_SECONDS)
        self.frame_sample_rate = config.FRAME_SAMPLE_RATE

    def load(self, video_path: str, deadline: float) -> Tuple[Iterator[np.ndarray], Dict[str, Any]]:
        """Load, validate, and yield sampled frames from a video file.

        Args:
            video_path: Path to the video file.
            deadline: time.monotonic() timestamp for the global timeout.

        Returns:
            A tuple of (frames_iterator, metadata).
                - frames_iterator: Iterator yielding sampled np.ndarray frames (BGR).
                - metadata: Dictionary containing stream metadata.

        Raises:
            InvalidVideoFormatError: If the extension is unsupported.
            UploadSizeExceededError: If the file size or duration is too large.
            VideoCorruptionError: If the video cannot be opened or metadata is invalid.
        """
        logger.info("[FFMPEG_INGEST] Starting ingestion for: %s", video_path)
        path = Path(video_path)

        if not path.exists():
            raise VideoCorruptionError(f"File not found: {path}")

        self._validate_extension(path)

        size_bytes = os.path.getsize(path)
        if size_bytes == 0:
            raise VideoCorruptionError(f"File is empty: {path}")

        if not os.access(path, os.R_OK):
            raise VideoCorruptionError(f"File is not readable: {path}")

        self._validate_file_size(path, size_bytes)
        logger.info("[FFMPEG_INGEST] Input file size valid (%d bytes).", size_bytes)

        metadata = self._extract_metadata_ffprobe(path, deadline)
        logger.info("[FFMPEG_INGEST] Metadata: %s", metadata)

        self._validate_duration(metadata["duration_seconds"], path)

        frames_iterator = self._sample_frames_ffmpeg(path, metadata, deadline)
        
        logger.info(
            "[FFMPEG_INGEST] Video metadata complete, returning streaming iterator for %s (%.1fs @ %.1f fps)",
            path.name,
            metadata["duration_seconds"],
            metadata["fps"],
        )
        return frames_iterator, metadata

    def _validate_extension(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidVideoFormatError(
                f"Unsupported file type '{ext}'. "
                f"Please upload an MP4 or AVI video."
            )

    def _validate_file_size(self, path: Path, size_bytes: int) -> None:
        if size_bytes > self.max_file_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            raise UploadSizeExceededError(
                f"File size {size_mb:.1f} MB exceeds the maximum allowed "
                f"{config.MAX_FILE_SIZE_MB} MB."
            )

    def _validate_duration(self, duration_seconds: float, path: Path) -> None:
        if duration_seconds > self.max_duration_seconds:
            raise UploadSizeExceededError(
                f"Video duration {duration_seconds:.1f}s exceeds the maximum "
                f"allowed {config.MAX_DURATION_SECONDS}s."
            )

    def _extract_metadata_ffprobe(self, path: Path, deadline: float) -> Dict[str, Any]:
        """Extract stream metadata using ffprobe."""
        if not shutil.which("ffprobe"):
            raise VideoCorruptionError("ffprobe executable not found in PATH")

        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path)
        ]

        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise VideoCorruptionError("Global deadline exceeded before starting FFprobe.")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                stderr_trunc = result.stderr[:8192] if result.stderr else ""
                raise VideoCorruptionError(f"ffprobe failed (exit {result.returncode}): {stderr_trunc}")
                
            probe_data = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise VideoCorruptionError("ffprobe timed out extracting metadata")
        except json.JSONDecodeError:
            raise VideoCorruptionError("ffprobe output was not valid JSON")

        streams = probe_data.get("streams", [])
        if not streams:
            raise VideoCorruptionError("No video streams found in file")

        stream = streams[0]
        format_info = probe_data.get("format", {})

        width = stream.get("width", 0)
        height = stream.get("height", 0)

        # Parse FPS
        fps = 0.0
        avg_frame_rate = stream.get("avg_frame_rate", "0/0")
        if "/" in avg_frame_rate:
            num, den = avg_frame_rate.split("/")
            if int(den) > 0:
                fps = float(num) / float(den)
        else:
            try:
                fps = float(avg_frame_rate)
            except ValueError:
                pass

        # Parse duration
        duration_str = stream.get("duration") or format_info.get("duration")
        duration = float(duration_str) if duration_str else 0.0

        # Parse frame count (if available, else derive from duration and fps)
        frame_count_str = stream.get("nb_frames")
        if frame_count_str and frame_count_str != "N/A":
            frame_count = int(frame_count_str)
        else:
            if duration > 0 and fps > 0:
                frame_count = int(duration * fps)
            else:
                frame_count = 0

        issues = []
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

    def _sample_frames_ffmpeg(self, path: Path, metadata: Dict[str, Any], deadline: float) -> Iterator[np.ndarray]:
        """Extract and yield sampled frames using an ffmpeg subprocess.
        
        Reads exactly width*height*3 bytes per frame from stdout pipe.
        Yields frames lazily to maintain O(1) memory bound.
        """
        if not shutil.which("ffmpeg"):
            raise VideoCorruptionError("ffmpeg executable not found in PATH")

        width = metadata["width"]
        height = metadata["height"]
        frame_bytes = width * height * 3
        sample_rate = metadata["sample_rate"]

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]
        
        logger.info("[FFMPEG_INGEST] Command/configuration: ffmpeg -i <file> -f image2pipe -pix_fmt bgr24")
        
        stderr_file = tempfile.TemporaryFile()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_file, stdin=subprocess.DEVNULL
        )
        logger.info("[FFMPEG_INGEST] Process started (PID: %d)", process.pid)
        logger.info("[FFMPEG_INGEST] Waiting for first frame bytes...")

        frame_index = 0

        try:
            while True:
                if time.monotonic() > deadline:
                    raise VideoCorruptionError("Global deadline exceeded during FFmpeg decode")

                raw_frame = bytearray()
                bytes_needed = frame_bytes
                while bytes_needed > 0:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise VideoCorruptionError("Global deadline exceeded waiting for FFmpeg stdout")
                    
                    if sys.platform != "win32":
                        import select
                        r, _, _ = select.select([process.stdout.fileno()], [], [], timeout)
                        if not r:
                            raise VideoCorruptionError("Read timeout waiting for FFmpeg stdout")
                            
                    chunk = process.stdout.read1(min(bytes_needed, 1048576))
                    if not chunk:
                        break
                    raw_frame.extend(chunk)
                    bytes_needed -= len(chunk)
                
                raw_frame = bytes(raw_frame)
                
                if not raw_frame:
                    break # EOF
                    
                if len(raw_frame) != frame_bytes:
                    raise VideoCorruptionError(f"Partial frame read: {len(raw_frame)} / {frame_bytes} bytes. Subprocess may have crashed.")

                if frame_index == 0:
                    logger.info("[FFMPEG_INGEST] First complete raw frame received (size: %d bytes)", len(raw_frame))

                if frame_index % sample_rate == 0:
                    frame_array = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    frame_array = frame_array.copy()
                    
                    if frame_index == 0:
                        logger.info("[FFMPEG_INGEST] First sampled frame yielded")
                    elif frame_index % (sample_rate * 50) == 0:
                        logger.info("[FFMPEG_INGEST] Yielding sampled frame index %d", frame_index)
                        
                    yield frame_array
                
                # raw_frame buffer is discarded, memory reclaimed
                del raw_frame
                frame_index += 1

        finally:
            # Clean up subprocess guarantees
            logger.info("[FFMPEG_INGEST] Decoder completed iteration, starting cleanup")
            try:
                stderr_tail = ""
                if stderr_file:
                    stderr_file.seek(0)
                    stderr_tail = stderr_file.read().decode(errors="replace")
                    stderr_file.close()
                if process.stdout:
                    process.stdout.close()
                
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                else:
                    process.wait()
            except Exception as e:
                logger.error("Error closing pipes or waiting for process: %s", e)
                process.kill()
                process.wait()

            logger.info("[FFMPEG_INGEST] Process reaped. Return code: %s", process.returncode)

            if process.returncode != 0 and process.returncode is not None:
                stderr_trunc = stderr_tail[:8192] if stderr_tail else ""
                if process.returncode == -15 or process.returncode == -9: # SIGTERM or SIGKILL
                    raise VideoCorruptionError(f"FFmpeg was killed or terminated. stderr: {stderr_trunc}")
                raise VideoCorruptionError(f"FFmpeg failed with exit code {process.returncode}. stderr: {stderr_trunc}")

