import argparse
import hashlib
import json
import logging
import mimetypes
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [RENDER_DIAGNOSTIC] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = os.path.join(tempfile.gettempdir(), "crowddna_diagnostic_checkpoint.json")

def get_rss_mb():
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

def write_checkpoint(stage, rss_mb, duration_seconds, status="PASS"):
    try:
        data = {
            "stage": stage,
            "timestamp": time.time(),
            "rss_mb": rss_mb,
            "duration_seconds": duration_seconds,
            "status": status
        }
        tmp_path = CHECKPOINT_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, CHECKPOINT_FILE)
    except Exception as e:
        logger.error(f"Failed to write checkpoint: {e}")

class StageTracker:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.start_rss = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_rss = get_rss_mb()
        if self.start_rss is not None:
            logger.info(f"[{self.name}] RSS before: {self.start_rss:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        end_rss = get_rss_mb()
        
        if end_rss is not None:
            logger.info(f"[{self.name}] RSS after: {end_rss:.2f} MB")
            if self.start_rss is not None:
                delta = end_rss - self.start_rss
                logger.info(f"[{self.name}] RSS delta: {delta:.2f} MB")
        
        logger.info(f"[{self.name}] Duration: {duration:.4f} seconds")
        
        status = "FAIL" if exc_type else "PASS"
        if exc_type:
            logger.error(f"[{self.name}] Stage failed with {exc_type.__name__}: {exc_val}")
            
        write_checkpoint(self.name, end_rss, duration, status)
        return False # Do not suppress exceptions

def run_render_upload_diagnostic(video_path: str):
    logger.info(f"Starting diagnostic for: {video_path}")
    
    # Clean up stale checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception as e:
            logger.error(f"Could not remove stale checkpoint: {e}")

    results = {
        "ffprobe": "SKIPPED",
        "ffmpeg": "SKIPPED",
        "opencv_open": "FAIL",
        "opencv_metadata": "FAIL",
        "opencv_read": "FAIL",
        "opencv_release": "FAIL"
    }
    
    peak_rss = get_rss_mb() or 0.0
    baseline_rss = peak_rss
    
    def update_peak():
        nonlocal peak_rss
        curr = get_rss_mb()
        if curr and curr > peak_rss:
            peak_rss = curr

    file_size = None
    sha256_hex = None

    try:
        # Stage A
        with StageTracker("BASELINE"):
            import cv2
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Platform: {platform.platform()}")
            logger.info(f"OpenCV version: {cv2.__version__}")
            
            build_info = cv2.getBuildInformation()
            has_ffmpeg = "FFMPEG" in build_info.upper()
            logger.info(f"FFMPEG support in OpenCV: {has_ffmpeg}")
            logger.info(f"Process PID: {os.getpid()}")
            logger.info(f"Thread count: {threading.active_count()}")
            
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            logger.info(f"MemAvailable: {line.split()[1]} kB")
                            break
            except Exception:
                pass
        update_peak()

        # Stage B
        with StageTracker("FILE_INSPECTION"):
            abs_path = os.path.abspath(video_path)
            logger.info(f"Absolute path: {abs_path}")
            
            if not os.path.exists(abs_path):
                logger.error(f"File does not exist: {abs_path}")
                raise FileNotFoundError(f"File not found: {abs_path}")
            
            file_size = os.path.getsize(abs_path)
            logger.info(f"File size: {file_size} bytes")
            
            _, ext = os.path.splitext(abs_path)
            logger.info(f"Extension: {ext}")
            
            mime, _ = mimetypes.guess_type(abs_path)
            logger.info(f"MIME type: {mime}")
            
            perms = oct(os.stat(abs_path).st_mode)[-3:]
            logger.info(f"Permissions: {perms}")
            
            sha256_hash = hashlib.sha256()
            with open(abs_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            sha256_hex = sha256_hash.hexdigest()
            logger.info(f"SHA256: {sha256_hex}")
        update_peak()

        # Stage C
        with StageTracker("FFPROBE"):
            try:
                cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-select_streams", "v:0", 
                    "-show_entries", "stream=codec_name,codec_long_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration,bit_rate", 
                    "-show_entries", "format=format_name,duration,bit_rate",
                    "-of", "json",
                    video_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                stdout_trunc = result.stdout[:8192] if result.stdout else ""
                stderr_trunc = result.stderr[:8192] if result.stderr else ""
                
                if result.returncode == 0:
                    results["ffprobe"] = "PASS"
                    logger.info("ffprobe PASS")
                    try:
                        probe_data = json.loads(stdout_trunc)
                        logger.info(f"ffprobe data: {json.dumps(probe_data)}")
                    except Exception as e:
                        logger.error(f"Failed to parse ffprobe json: {e}")
                else:
                    results["ffprobe"] = "FAIL"
                    logger.error(f"ffprobe failed (exit code {result.returncode})")
                    logger.error(f"stderr: {stderr_trunc}")
            except FileNotFoundError:
                logger.warning("ffprobe not found in PATH")
            except subprocess.TimeoutExpired:
                results["ffprobe"] = "FAIL"
                logger.error("ffprobe timed out after 15s")
        update_peak()

        # Stage D
        with StageTracker("FFMPEG_DECODE"):
            out_img = os.path.join(tempfile.gettempdir(), f"diagnostic_frame_{os.getpid()}.jpg")
            try:
                if os.path.exists(out_img):
                    os.remove(out_img)
                    
                cmd = [
                    "ffmpeg",
                    "-v", "error",
                    "-y",
                    "-i", video_path,
                    "-frames:v", "1",
                    out_img
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                
                stderr_trunc = result.stderr[:8192] if result.stderr else ""
                
                if result.returncode == 0 and os.path.exists(out_img):
                    results["ffmpeg"] = "PASS"
                    logger.info("FFmpeg extracted 1 frame successfully.")
                    logger.info(f"Output size: {os.path.getsize(out_img)} bytes")
                else:
                    results["ffmpeg"] = "FAIL"
                    logger.error(f"FFmpeg failed (exit code {result.returncode})")
                    logger.error(f"stderr: {stderr_trunc}")
            except FileNotFoundError:
                logger.warning("ffmpeg not found in PATH")
            except subprocess.TimeoutExpired:
                results["ffmpeg"] = "FAIL"
                logger.error("ffmpeg timed out after 15s")
            finally:
                if os.path.exists(out_img):
                    try:
                        os.remove(out_img)
                    except Exception:
                        pass
        update_peak()

        # Stage E
        cap = None
        with StageTracker("OPENCV_OPEN"):
            import cv2
            write_checkpoint("OPENCV_OPEN_START", get_rss_mb(), 0, "IN_PROGRESS")
            
            t0 = time.time()
            cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
            t1 = time.time()
            
            logger.info(f"Time inside VideoCapture constructor: {t1 - t0:.4f}s")
            
            is_opened = cap.isOpened()
            logger.info(f"isOpened(): {is_opened}")
            
            if is_opened:
                results["opencv_open"] = "PASS"
                logger.info(f"backend: {cap.getBackendName()}")
            else:
                results["opencv_open"] = "FAIL"
                raise RuntimeError("OpenCV VideoCapture failed to open file.")
        update_peak()

        # Stage F
        with StageTracker("OPENCV_METADATA"):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            
            logger.info(f"FPS: {fps}")
            logger.info(f"Frame count: {frame_count}")
            logger.info(f"Width: {width}")
            logger.info(f"Height: {height}")
            logger.info(f"FOURCC: {fourcc}")
            results["opencv_metadata"] = "PASS"
        update_peak()

        # Stage G
        with StageTracker("OPENCV_FIRST_READ"):
            write_checkpoint("OPENCV_FIRST_READ_START", get_rss_mb(), 0, "IN_PROGRESS")
            logger.info("[RENDER_DIAGNOSTIC] About to execute OpenCV first cap.read()")
            
            t0 = time.time()
            ret, frame = cap.read()
            t1 = time.time()
            
            logger.info(f"Time inside cap.read(): {t1 - t0:.4f}s")
            logger.info(f"return flag (ret): {ret}")
            
            if ret and frame is not None:
                results["opencv_read"] = "PASS"
                logger.info(f"frame shape: {frame.shape}")
            else:
                results["opencv_read"] = "FAIL"
                logger.error("cap.read() returned False or None frame")
        update_peak()

    except Exception as e:
        logger.error(f"Diagnostic interrupted by exception: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Stage H
        if 'cap' in locals() and cap is not None:
            try:
                with StageTracker("OPENCV_RELEASE"):
                    cap.release()
                    results["opencv_release"] = "PASS"
            except Exception as e:
                logger.error(f"Failed to release cap: {e}")
        update_peak()
        
        final_rss = get_rss_mb() or 0.0
        
        print("\n==================================================")
        print("CROWD DNA RENDER UPLOAD DIAGNOSTIC SUMMARY")
        print("==================================================")
        print("\nFILE")
        print(f"size: {file_size}")
        print(f"sha256: {sha256_hex}")
        print("\nMEMORY")
        print(f"baseline RSS: {baseline_rss:.2f} MB")
        print(f"final RSS: {final_rss:.2f} MB")
        print(f"peak observed RSS: {peak_rss:.2f} MB")
        print("\nFFPROBE")
        print(results["ffprobe"])
        print("\nFFMPEG")
        print(results["ffmpeg"])
        print("\nOPENCV OPEN")
        print(results["opencv_open"])
        print("\nOPENCV METADATA")
        print(results["opencv_metadata"])
        print("\nOPENCV FIRST READ")
        print(results["opencv_read"])
        print("\nOPENCV RELEASE")
        print(results["opencv_release"])
        print("\n==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Upload Diagnostic")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    run_render_upload_diagnostic(args.video_path)
