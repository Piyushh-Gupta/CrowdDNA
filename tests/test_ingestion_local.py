"""Local integration test for FFmpeg ingestion.

This creates a real synthetic MP4 using ffmpeg locally, 
then runs VideoIngestor.load() to verify the subprocess
extraction works end-to-end on this host.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna.ingestion.video_loader import VideoIngestor

def test_local_ingestion_end_to_end(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe not available, skipping local integration test.")
        return

    test_video = tmp_path / "synthetic_test.mp4"
    
    # Generate 1 second of 10 fps video (10 frames)
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", "testsrc=duration=1:size=320x240:rate=10",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(test_video)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    ingestor = VideoIngestor()
    frames, metadata = ingestor.load(str(test_video))

    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert metadata["fps"] == 10.0
    assert metadata["frame_count"] == 10

    expected_frames = sum(1 for i in range(10) if i % ingestor.frame_sample_rate == 0)
    assert len(frames) == expected_frames
    
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.shape == (240, 320, 3)

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_local_ingestion_end_to_end(Path(td))
        print("Local integration test passed.")
