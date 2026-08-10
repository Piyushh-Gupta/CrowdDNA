"""Video ingestion module for CrowdFlow DNA.

This module validates uploaded video files, extracts metadata using FFprobe,
and samples frames using an FFmpeg subprocess for downstream processing.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

        metadata = self._extract_metadata_ffprobe(path)
        logger.info("[FFMPEG_INGEST] Metadata: %s", metadata)

        self._validate_duration(metadata["duration_seconds"], path)

        frames = self._sample_frames_ffmpeg(path, metadata)
        
        logger.info(
            "[FFMPEG_INGEST] Cleanup complete. Ingestion complete: %d frames sampled from %s (%.1fs @ %.1f fps)",
            len(frames),
            path.name,
            metadata["duration_seconds"],
            metadata["fps"],
        )
        return frames, metadata

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

    def _extract_metadata_ffprobe(self, path: Path) -> Dict[str, Any]:
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

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
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

    def _sample_frames_ffmpeg(self, path: Path, metadata: Dict[str, Any]) -> List[np.ndarray]:
        """Extract and sample frames using an ffmpeg subprocess.
        
        Reads exactly width*height*3 bytes per frame from stdout pipe.
        Uses an independent watchdog thread to ensure we never block indefinitely.
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
        
        rss_start = get_rss_mb()
        if rss_start is not None:
            logger.info("[FFMPEG_INGEST] RSS before: %.2f MB", rss_start)

        t0 = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7
        )
        logger.info("[FFMPEG_INGEST] Process started (PID: %d)", process.pid)

        timeout_seconds = float(config.MAX_DURATION_SECONDS) + 30.0
        
        # Watchdog thread ensures process is killed if it hangs, breaking stdout.read()
        def watchdog():
            start = time.time()
            while process.poll() is None:
                if time.time() - start > timeout_seconds:
                    logger.warning("[FFMPEG_INGEST] Timeout detected")
                    logger.info("[FFMPEG_INGEST] Terminating process")
                    process.terminate()
                    time.sleep(1.0)
                    if process.poll() is None:
                        logger.info("[FFMPEG_INGEST] Killing process")
                        process.kill()
                    break
                time.sleep(0.5)

        wd_thread = threading.Thread(target=watchdog, daemon=True)
        wd_thread.start()

        frames = []
        frame_index = 0
        peak_rss = rss_start or 0.0

        try:
            while True:
                # Read exactly frame_bytes
                raw_frame = b''
                bytes_needed = frame_bytes
                while bytes_needed > 0:
                    chunk = process.stdout.read(bytes_needed)
                    if not chunk:
                        break
                    raw_frame += chunk
                    bytes_needed -= len(chunk)
                
                if not raw_frame:
                    break # EOF
                    
                if len(raw_frame) != frame_bytes:
                    raise VideoCorruptionError(f"Partial frame read: {len(raw_frame)} / {frame_bytes} bytes. Subprocess may have crashed.")

                curr_rss = get_rss_mb()
                if curr_rss and curr_rss > peak_rss:
                    peak_rss = curr_rss

                if frame_index % sample_rate == 0:
                    # Convert to numpy array and reshape
                    frame_array = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                    # Make a copy so we don't hold references to a huge contiguous buffer if the OS provides one,
                    # though frombuffer on a new bytes object is already decoupled. We just use it directly.
                    frame_array = frame_array.copy()
                    frames.append(frame_array)
                    
                    if len(frames) == 1:
                        logger.info("[FFMPEG_INGEST] First sampled frame received")
                    else:
                        logger.debug("[FFMPEG_INGEST] Sample frame received (idx: %d)", frame_index)
                
                # raw_frame buffer is discarded for non-sampled frames here, reclaiming memory
                del raw_frame
                frame_index += 1

        except Exception as e:
            logger.error("[FFMPEG_INGEST] Exception during frame extraction: %s", e)
            process.terminate()
            raise
        finally:
            # Ensure stdout/stderr are closed and process is reaped
            try:
                stderr_tail = process.stderr.read(8192).decode(errors="replace") if process.stderr else ""
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                process.wait(timeout=5)
            except Exception as e:
                logger.error("Error closing pipes or waiting for process: %s", e)
                process.kill()
                process.wait()

            logger.info("[FFMPEG_INGEST] Process reaped")

        t1 = time.time()
        logger.info("[FFMPEG_INGEST] Process completed in %.2f seconds", (t1 - t0))
        logger.info("[FFMPEG_INGEST] Process return code: %s", process.returncode)

        if process.returncode != 0:
            stderr_trunc = stderr_tail[:8192] if stderr_tail else ""
            if process.returncode == -15 or process.returncode == -9: # SIGTERM or SIGKILL
                raise VideoCorruptionError(f"FFmpeg timed out and was killed. stderr: {stderr_trunc}")
            raise VideoCorruptionError(f"FFmpeg failed with exit code {process.returncode}. stderr: {stderr_trunc}")

        rss_end = get_rss_mb()
        if rss_end is not None:
            logger.info("[FFMPEG_INGEST] RSS after: %.2f MB", rss_end)
            if rss_start is not None:
                logger.info("[FFMPEG_INGEST] RSS delta: %.2f MB", rss_end - rss_start)
            logger.info("[FFMPEG_INGEST] observed peak RSS: %.2f MB", peak_rss)

        if len(frames) == 0:
            raise VideoCorruptionError("FFmpeg extracted 0 valid frames.")

        return frames
