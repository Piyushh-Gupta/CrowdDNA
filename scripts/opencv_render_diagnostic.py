import argparse
import concurrent.futures
import logging
import os
import platform
import sys
import time
import traceback

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def log_environment_info():
    logger.info("=== ENVIRONMENT INFO ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"CWD: {os.getcwd()}")
    logger.info("OpenCV Build Information:")
    logger.info(cv2.getBuildInformation())
    logger.info("========================")

def test_video(path: str) -> None:
    logger.info(f"\n--- Testing Video: {path} ---")
    
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return

    logger.info(f"File size: {os.path.getsize(path)} bytes")
    logger.info("Opening...")
    
    cap = None
    try:
        # Match production behavior: force FFMPEG
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        
        is_opened = cap.isOpened()
        logger.info(f"Opened?: {is_opened}")
        if not is_opened:
            logger.error("Failed to open VideoCapture.")
            return

        backend_name = cap.getBackendName()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        logger.info(f"Backend name: {backend_name}")
        logger.info(f"FPS: {fps}")
        logger.info(f"Frame count: {frame_count}")
        logger.info(f"Width: {width}")
        logger.info(f"Height: {height}")
        logger.info(f"FOURCC Codec: {codec}")

        logger.info("Executing cap.read() with 10 second timeout...")
        start_time = time.time()
        
        # Test the direct block to avoid thread pool complexity for the isolated diagnostic, 
        # wait, the issue is that it hangs inside cap.read() which blocks the main thread.
        # But if we want to timeout in the diagnostic, we DO need a thread pool.
        
        def _read_worker():
            return cap.read()
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_read_worker)
            try:
                ret, frame = future.result(timeout=10.0)
                elapsed = time.time() - start_time
                
                logger.info("SUCCESS")
                logger.info(f"Read time: {elapsed:.4f}s")
                logger.info(f"Return flag: {ret}")
                if frame is not None:
                    logger.info(f"Frame shape: {frame.shape}")
                else:
                    logger.info("Frame is None")
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                logger.error("TIMEOUT")
                logger.error(f"cap.read() hung for {elapsed:.4f}s.")
            except Exception:
                logger.error("EXCEPTION during cap.read()")
                logger.error(traceback.format_exc())
                
    except Exception:
        logger.error("EXCEPTION during initialization")
        logger.error(traceback.format_exc())
    finally:
        if cap is not None:
            logger.info("Calling cap.release()...")
            cap.release()
            logger.info("cap.release() complete.")
    logger.info("----------------------------------\n")

def create_synthetic_video(path: str = "synthetic.mp4") -> None:
    logger.info(f"Generating synthetic MP4 at {path}...")
    width, height = 320, 240
    fps = 30.0
    frame_count = 30
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Failed to open VideoWriter for {path}")
        return

    try:
        for i in range(frame_count):
            # Fill with a solid color changing over time
            frame = np.full((height, width, 3), (0, i * 8, 255 - i * 8), dtype=np.uint8)
            out.write(frame)
        logger.info("Synthetic video generated successfully.")
    except Exception:
        logger.error("Failed to write synthetic video")
        logger.error(traceback.format_exc())
    finally:
        out.release()

def main():
    parser = argparse.ArgumentParser(description="OpenCV Render Diagnostic Utility")
    parser.add_argument("video_path", nargs="?", help="Optional path to a video to test.")
    args = parser.parse_args()

    log_environment_info()

    if args.video_path:
        test_video(args.video_path)
    else:
        # Default behavior: generate and test synthetic video
        synthetic_path = "synthetic.mp4"
        create_synthetic_video(synthetic_path)
        test_video(synthetic_path)

if __name__ == "__main__":
    main()
