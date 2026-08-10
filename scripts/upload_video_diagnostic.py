import argparse
import hashlib
import json
import logging
import mimetypes
import os
import subprocess
import sys
import time
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [UPLOAD_DIAGNOSTIC] %(message)s")
logger = logging.getLogger(__name__)

def get_rss_mb() -> float:
    """Returns the current RSS memory usage in MB."""
    try:
        with open('/proc/self/statm', 'r') as f:
            pages = int(f.read().split()[1])
            page_size = os.sysconf('SC_PAGE_SIZE')
            return (pages * page_size) / (1024 * 1024)
    except Exception:
        pass
    
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return 0.0

def log_memory(stage: str):
    rss = get_rss_mb()
    logger.info(f"Memory (RSS) before {stage}: {rss:.2f} MB")
    return rss

def run_diagnostic(video_path: str) -> dict:
    start_rss = log_memory("imports")
    
    logger.info(f"Starting diagnostic for: {video_path}")
    results = {
        "ffprobe": "FAIL",
        "ffmpeg": "FAIL",
        "opencv_capture": "FAIL",
        "opencv_read": "FAIL"
    }
    peak_rss = start_rss
    
    # ---------------------------------------------------------
    # Stage 1 — File Integrity
    # ---------------------------------------------------------
    logger.info("--- Stage 1: File Integrity ---")
    try:
        abs_path = os.path.abspath(video_path)
        logger.info(f"Absolute path: {abs_path}")
        
        if not os.path.exists(abs_path):
            logger.error(f"File does not exist: {abs_path}")
            return results

        size = os.path.getsize(abs_path)
        logger.info(f"File size: {size} bytes")
        
        _, ext = os.path.splitext(abs_path)
        logger.info(f"Extension: {ext}")
        
        mime, _ = mimetypes.guess_type(abs_path)
        logger.info(f"MIME type: {mime}")
        
        perms = oct(os.stat(abs_path).st_mode)[-3:]
        logger.info(f"File permissions: {perms}")
        
        sha256_hash = hashlib.sha256()
        with open(abs_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        logger.info(f"SHA256 checksum: {sha256_hash.hexdigest()}")
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        logger.error(traceback.format_exc())

    peak_rss = max(peak_rss, log_memory("ffprobe"))

    # ---------------------------------------------------------
    # Stage 2 — ffprobe
    # ---------------------------------------------------------
    logger.info("--- Stage 2: ffprobe ---")
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,duration", 
            "-of", "json",
            video_path
        ]
        start_time = time.time()
        # Limit stdout/stderr capture to 16KB to save memory
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        stdout_trunc = result.stdout[:16384] if result.stdout else ""
        stderr_trunc = result.stderr[:16384] if result.stderr else ""
        
        logger.info(f"ffprobe exit code: {result.returncode} (took {elapsed:.4f}s)")
        if result.returncode == 0:
            results["ffprobe"] = "PASS"
            probe_data = json.loads(stdout_trunc)
            stream_info = probe_data.get("streams", [{}])[0]
            
            logger.info(f"Codec: {stream_info.get('codec_name')}")
            logger.info(f"Duration: {stream_info.get('duration')}s")
            logger.info(f"FPS: {stream_info.get('avg_frame_rate')}")
            logger.info(f"Width: {stream_info.get('width')}")
            logger.info(f"Height: {stream_info.get('height')}")
        else:
            logger.error(f"ffprobe failed. stdout: {stdout_trunc}")
            logger.error(f"ffprobe failed. stderr: {stderr_trunc}")
    except FileNotFoundError:
        logger.error("ffprobe executable not found in PATH.")
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        logger.error(traceback.format_exc())

    peak_rss = max(peak_rss, log_memory("ffmpeg"))

    # ---------------------------------------------------------
    # Stage 3 — FFmpeg Decode Test
    # ---------------------------------------------------------
    logger.info("--- Stage 3: FFmpeg Decode Test ---")
    out_img = os.path.join(os.path.dirname(abs_path), "diagnostic_frame.jpg")
    try:
        if os.path.exists(out_img):
            os.remove(out_img)
            
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-y", # overwrite output
            "-i", video_path,
            "-frames:v", "1",
            out_img
        ]
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        stdout_trunc = result.stdout[:16384] if result.stdout else ""
        stderr_trunc = result.stderr[:16384] if result.stderr else ""
        
        logger.info(f"ffmpeg exit code: {result.returncode} (took {elapsed:.4f}s)")
        if result.returncode == 0 and os.path.exists(out_img):
            results["ffmpeg"] = "PASS"
            logger.info("Frame extracted successfully.")
            logger.info(f"Output image exists: {out_img}")
            logger.info(f"Output image size: {os.path.getsize(out_img)} bytes")
        else:
            logger.error(f"ffmpeg failed. stdout: {stdout_trunc}")
            logger.error(f"ffmpeg failed. stderr: {stderr_trunc}")
            if not os.path.exists(out_img):
                logger.error(f"Output image was not created: {out_img}")
    except FileNotFoundError:
        logger.error("ffmpeg executable not found in PATH.")
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        if os.path.exists(out_img):
            try:
                os.remove(out_img)
            except Exception:
                pass

    peak_rss = max(peak_rss, log_memory("OpenCV"))

    # ---------------------------------------------------------
    # Stage 4 — OpenCV Decode Test
    # ---------------------------------------------------------
    logger.info("--- Stage 4: OpenCV Decode Test ---")
    cap = None
    try:
        # Lazy import OpenCV to save memory until absolutely needed
        import cv2
        
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        start_time = time.time()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        elapsed_cap = time.time() - start_time
        logger.info(f"time to VideoCapture(): {elapsed_cap:.4f}s")
        
        is_opened = cap.isOpened()
        logger.info(f"isOpened(): {is_opened}")
        
        if is_opened:
            results["opencv_capture"] = "PASS"
            logger.info(f"backend: {cap.getBackendName()}")
            
            logger.info(f"metadata: FPS={cap.get(cv2.CAP_PROP_FPS)}, "
                        f"W={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
                        f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, "
                        f"FOURCC={cap.get(cv2.CAP_PROP_FOURCC)}")
            
            start_time = time.time()
            ret, frame = cap.read()
            elapsed_read = time.time() - start_time
            logger.info(f"time to first cap.read(): {elapsed_read:.4f}s")
            logger.info(f"return flag: {ret}")
            
            if ret and frame is not None:
                results["opencv_read"] = "PASS"
                logger.info(f"frame shape: {frame.shape}")
            else:
                logger.error("Failed to read frame or frame is None.")
        else:
            logger.error("VideoCapture failed to open.")
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        if cap is not None:
            logger.info("release()...")
            cap.release()
            logger.info("release() complete.")

    final_rss = log_memory("summary")
    peak_rss = max(peak_rss, final_rss)

    # ---------------------------------------------------------
    # Stage 5 — Compare Results
    # ---------------------------------------------------------
    logger.info("========== SUMMARY ==========")
    logger.info(f"ffprobe:             {results['ffprobe']}")
    logger.info(f"FFmpeg decode:       {results['ffmpeg']}")
    logger.info(f"OpenCV VideoCapture: {results['opencv_capture']}")
    logger.info(f"OpenCV first read:   {results['opencv_read']}")
    logger.info(f"Peak RSS Memory:     {peak_rss:.2f} MB")
    logger.info("=============================")

    return results


def main():
    parser = argparse.ArgumentParser(description="Upload Video Diagnostic")
    parser.add_argument("video_path", help="Path to a video to test.")
    args = parser.parse_args()
    
    run_diagnostic(args.video_path)

if __name__ == "__main__":
    main()
