import argparse
import logging
import os
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MINIMAL_DIAG] %(message)s")
logger = logging.getLogger(__name__)

def run_minimal_diagnostic(video_path):
    logger.info("1. Python execution started")
    
    logger.info("Before checking file access: %s", video_path)
    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        logger.info("2. File access OK, size: %d bytes", size)
    else:
        logger.error("2. File access FAILED (not found)")
        return
        
    logger.info("Before shutil.which('ffprobe')")
    ffprobe_path = shutil.which("ffprobe")
    logger.info("3. ffprobe availability: %s", ffprobe_path)
    
    if ffprobe_path:
        logger.info("Before ffprobe process creation")
        try:
            # We use timeout=15 to ensure it doesn't hang forever
            cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", video_path]
            logger.info("4. ffprobe process creation starting: %s", cmd)
            
            logger.info("Before subprocess.run(ffprobe)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            logger.info("After subprocess.run(ffprobe)")
            
            logger.info("5. ffprobe execution finished with return code: %d", result.returncode)
            if result.stdout:
                logger.info("ffprobe stdout (trunc): %s", result.stdout[:200])
            if result.stderr:
                logger.warning("ffprobe stderr (trunc): %s", result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.error("5. ffprobe execution TIMEOUT")
        except Exception as e:
            logger.error("5. ffprobe execution Exception: %s", e)
            
    logger.info("Before shutil.which('ffmpeg')")
    ffmpeg_path = shutil.which("ffmpeg")
    logger.info("6. ffmpeg availability: %s", ffmpeg_path)
    
    if ffmpeg_path:
        logger.info("Before ffmpeg process creation")
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-frames:v", "1", "-f", "null", "-"]
            logger.info("7. ffmpeg process creation starting: %s", cmd)
            
            logger.info("Before subprocess.run(ffmpeg)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            logger.info("After subprocess.run(ffmpeg)")
            
            logger.info("8. ffmpeg execution finished with return code: %d", result.returncode)
            if result.stderr:
                logger.info("ffmpeg stderr (trunc): %s", result.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.error("8. ffmpeg execution TIMEOUT")
        except Exception as e:
            logger.error("8. ffmpeg execution Exception: %s", e)
            
    logger.info("Minimal diagnostic completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()
    
    # Add a global timeout in case anything hangs uncontrollably outside subprocess calls
    import threading
    import _thread
    import time
    def force_exit():
        time.sleep(45)
        logger.error("GLOBAL TIMEOUT REACHED (45s). Forcing hard exit.")
        _thread.interrupt_main()
        os._exit(1)
        
    threading.Thread(target=force_exit, daemon=True).start()
    
    logger.info("Before run_minimal_diagnostic")
    run_minimal_diagnostic(args.video_path)
    logger.info("After run_minimal_diagnostic")
