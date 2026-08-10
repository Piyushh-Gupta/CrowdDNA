import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

def run_subprocess_with_timeout(cmd, timeout_seconds):
    """Run a subprocess using Popen with a strict watchdog timeout."""
    t0 = time.time()
    rss_before = get_rss_mb()
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    timeout_event = threading.Event()
    
    def watchdog():
        start = time.time()
        while process.poll() is None:
            if time.time() - start > timeout_seconds:
                timeout_event.set()
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(0.1)

    wd_thread = threading.Thread(target=watchdog, daemon=True)
    wd_thread.start()
    
    stdout_data, stderr_data = b"", b""
    try:
        stdout_data, stderr_data = process.communicate()
    except Exception as e:
        logger.error("[ROOT_CAUSE_DIAGNOSTIC] Exception during communicate: %s", e)
        process.terminate()
        process.wait(timeout=2)
        
    wd_thread.join(timeout=1.0)
    
    if process.poll() is None:
        process.kill()
        process.wait()
        
    t1 = time.time()
    rss_after = get_rss_mb()
    
    status = "PASS"
    if timeout_event.is_set():
        status = "TIMEOUT"
    elif process.returncode != 0:
        status = "FAIL"
        
    stdout_str = stdout_data.decode(errors="replace")
    stderr_str = stderr_data.decode(errors="replace")
    
    # Truncate
    stdout_trunc = stdout_str[:8192] if stdout_str else ""
    stderr_trunc = stderr_str[:8192] if stderr_str else ""
    
    return {
        "status": status,
        "returncode": process.returncode,
        "elapsed": t1 - t0,
        "rss_before": rss_before,
        "rss_after": rss_after,
        "stdout": stdout_trunc,
        "stderr": stderr_trunc
    }

def generate_synthetic_mp4(path):
    if not shutil.which("ffmpeg"):
        return False
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def test_ffprobe(video_path):
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", video_path]
    return run_subprocess_with_timeout(cmd, 10.0)

def test_ffmpeg(video_path):
    cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-frames:v", "1", "-f", "null", "-"]
    return run_subprocess_with_timeout(cmd, 15.0)

def classify_root_cause(tools_available, ffprobe_synth, ffprobe_upload, ffmpeg_synth, ffmpeg_upload):
    if not tools_available:
        return "TOOLS_MISSING"
    
    if ffprobe_upload == "TIMEOUT":
        if ffprobe_synth == "PASS":
            return "FFPROBE_UPLOAD_TIMEOUT"
        else:
            return "FFPROBE_GLOBAL_TIMEOUT"
            
    if ffprobe_upload == "FAIL":
        if ffprobe_synth == "PASS":
            return "FFPROBE_UPLOAD_FAILURE"
        else:
            return "FFPROBE_SYNTHETIC_FAILURE"

    if ffmpeg_upload == "TIMEOUT":
        if ffmpeg_synth == "PASS":
            return "FFMPEG_UPLOAD_TIMEOUT"
        else:
            return "FFMPEG_GLOBAL_TIMEOUT"
            
    if ffmpeg_upload == "FAIL":
        if ffmpeg_synth == "PASS":
            return "FFMPEG_UPLOAD_FAILURE"
        else:
            return "FFMPEG_SYNTHETIC_FAILURE"

    if ffprobe_upload == "PASS" and ffmpeg_upload == "PASS":
        return "BOTH_TOOLS_PASS_UPLOAD"

    return "INCONCLUSIVE"

def run_diagnostic(uploaded_video_path):
    logger.info("[ROOT_CAUSE_DIAGNOSTIC] START")
    logger.info("[ROOT_CAUSE_DIAGNOSTIC] TOOL_DISCOVERY")
    
    logger.info("Python version: %s", sys.version)
    logger.info("Platform: %s", platform.platform())
    logger.info("PID: %d", os.getpid())
    
    rss = get_rss_mb()
    if rss is not None:
        logger.info("Current RSS: %.2f MB", rss)
        
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    logger.info("Available memory: %s", line.strip())
                    break
    except Exception:
        pass
        
    ffprobe_path = shutil.which("ffprobe")
    ffmpeg_path = shutil.which("ffmpeg")
    logger.info("shutil.which('ffprobe'): %s", ffprobe_path)
    logger.info("shutil.which('ffmpeg'): %s", ffmpeg_path)
    
    tools_available = bool(ffprobe_path and ffmpeg_path)
    
    if ffprobe_path:
        res = subprocess.run([ffprobe_path, "-version"], capture_output=True, text=True)
        logger.info("ffprobe version:\n%s", res.stdout[:200])
    if ffmpeg_path:
        res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        logger.info("ffmpeg version:\n%s", res.stdout[:200])

    synthetic_video_path = "/tmp/synthetic_diag.mp4"
    if os.name == "nt":
        import tempfile
        synthetic_video_path = os.path.join(tempfile.gettempdir(), "synthetic_diag.mp4")
        
    generated = generate_synthetic_mp4(synthetic_video_path)
    logger.info("Synthetic video generated: %s", generated)
    
    # Defaults
    ffprobe_synth_status = "SKIPPED"
    ffprobe_upload_status = "SKIPPED"
    ffmpeg_synth_status = "SKIPPED"
    ffmpeg_upload_status = "SKIPPED"

    if tools_available:
        logger.info("[ROOT_CAUSE_DIAGNOSTIC] FFPROBE_SYNTHETIC")
        if generated:
            res = test_ffprobe(synthetic_video_path)
            ffprobe_synth_status = res["status"]
            logger.info("Status: %s, Time: %.2fs, Code: %s", res["status"], res["elapsed"], res["returncode"])
        
        logger.info("[ROOT_CAUSE_DIAGNOSTIC] FFPROBE_UPLOAD")
        res = test_ffprobe(uploaded_video_path)
        ffprobe_upload_status = res["status"]
        logger.info("Status: %s, Time: %.2fs, Code: %s", res["status"], res["elapsed"], res["returncode"])
        if res["stderr"]:
            logger.info("stderr:\n%s", res["stderr"])
            
        logger.info("[ROOT_CAUSE_DIAGNOSTIC] FFMPEG_SYNTHETIC")
        if generated:
            res = test_ffmpeg(synthetic_video_path)
            ffmpeg_synth_status = res["status"]
            logger.info("Status: %s, Time: %.2fs, Code: %s", res["status"], res["elapsed"], res["returncode"])
            
        logger.info("[ROOT_CAUSE_DIAGNOSTIC] FFMPEG_UPLOAD")
        res = test_ffmpeg(uploaded_video_path)
        ffmpeg_upload_status = res["status"]
        logger.info("Status: %s, Time: %.2fs, Code: %s", res["status"], res["elapsed"], res["returncode"])
        if res["stderr"]:
            logger.info("stderr:\n%s", res["stderr"])

    logger.info("[ROOT_CAUSE_DIAGNOSTIC] CLASSIFICATION")
    
    classification = classify_root_cause(
        tools_available, ffprobe_synth_status, ffprobe_upload_status, ffmpeg_synth_status, ffmpeg_upload_status
    )
    
    print("\n==================================================")
    print("FFPROBE:")
    print(f"  synthetic: {ffprobe_synth_status}")
    print(f"  uploaded: {ffprobe_upload_status}")
    print("\nFFMPEG:")
    print(f"  synthetic: {ffmpeg_synth_status}")
    print(f"  uploaded: {ffmpeg_upload_status}")
    print("\nROOT CAUSE CLASSIFICATION:")
    print(f"  {classification}")
    print("==================================================\n")

    logger.info("[ROOT_CAUSE_DIAGNOSTIC] END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()
    run_diagnostic(args.video_path)
