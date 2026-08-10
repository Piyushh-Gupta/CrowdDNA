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
        ffprobe_path = shutil.which("ffprobe")
        ffmpeg_path = shutil.which("ffmpeg")
        logger.info(f"shutil.which('ffprobe'): {ffprobe_path}")
        logger.info(f"shutil.which('ffmpeg'): {ffmpeg_path}")
        
        if ffprobe_path:
            res = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
            logger.info(f"ffprobe -version:\n{res.stdout[:200]}")
        if ffmpeg_path:
            res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            logger.info(f"ffmpeg -version:\n{res.stdout[:200]}")

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
                        
                        streams = probe_data.get("streams", [])
                        if streams:
                            width = streams[0].get("width", 0)
                            height = streams[0].get("height", 0)
                            
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
            try:
                if not width or not height:
                    logger.warning("Width/height unknown, cannot test raw byte decode accurately.")
                else:
                    cmd = [
                        "ffmpeg",
                        "-v", "error",
                        "-i", video_path,
                        "-f", "image2pipe",
                        "-pix_fmt", "bgr24",
                        "-vcodec", "rawvideo",
                        "-"
                    ]
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    
                    frame_bytes = width * height * 3
                    
                    def watchdog():
                        start = time.time()
                        while process.poll() is None:
                            if time.time() - start > 15.0:
                                process.terminate()
                                time.sleep(0.5)
                                if process.poll() is None:
                                    process.kill()
                                break
                            time.sleep(0.1)

                    wd_thread = threading.Thread(target=watchdog, daemon=True)
                    wd_thread.start()
                    
                    raw_frame = b''
                    bytes_needed = frame_bytes
                    while bytes_needed > 0:
                        chunk = process.stdout.read(bytes_needed)
                        if not chunk:
                            break
                        raw_frame += chunk
                        bytes_needed -= len(chunk)
                        
                    stderr_tail = process.stderr.read(8192).decode(errors="replace") if process.stderr else ""
                    process.stdout.close()
                    process.stderr.close()
                    process.wait(timeout=5)
                    
                    if len(raw_frame) == frame_bytes:
                        results["ffmpeg"] = "PASS"
                        logger.info("FFmpeg extracted 1 frame successfully.")
                    else:
                        results["ffmpeg"] = "FAIL"
                        logger.error(f"Partial or no frame read: {len(raw_frame)} / {frame_bytes}")
                        logger.error(f"stderr: {stderr_tail}")
            except FileNotFoundError:
                logger.warning("ffmpeg not found in PATH")
            except Exception as e:
                logger.error(f"FFmpeg subprocess failed: {e}")
        update_peak()

    except Exception as e:
        logger.error(f"Diagnostic interrupted by exception: {e}")
        logger.error(traceback.format_exc())
    finally:
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
        print("\n==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Upload Diagnostic")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    run_render_upload_diagnostic(args.video_path)
