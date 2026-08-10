import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.render_upload_diagnostic import classify_root_cause


def test_classify_root_cause():
    # Tools missing
    assert classify_root_cause(False, "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED") == "TOOLS_MISSING"
    
    # FFprobe timeouts
    assert classify_root_cause(True, "PASS", "TIMEOUT", "PASS", "PASS") == "FFPROBE_UPLOAD_TIMEOUT"
    assert classify_root_cause(True, "TIMEOUT", "TIMEOUT", "PASS", "PASS") == "FFPROBE_GLOBAL_TIMEOUT"
    
    # FFprobe failures
    assert classify_root_cause(True, "PASS", "FAIL", "PASS", "PASS") == "FFPROBE_UPLOAD_FAILURE"
    assert classify_root_cause(True, "FAIL", "FAIL", "PASS", "PASS") == "FFPROBE_SYNTHETIC_FAILURE"
    
    # FFmpeg timeouts
    assert classify_root_cause(True, "PASS", "PASS", "PASS", "TIMEOUT") == "FFMPEG_UPLOAD_TIMEOUT"
    assert classify_root_cause(True, "PASS", "PASS", "TIMEOUT", "TIMEOUT") == "FFMPEG_GLOBAL_TIMEOUT"
    
    # FFmpeg failures
    assert classify_root_cause(True, "PASS", "PASS", "PASS", "FAIL") == "FFMPEG_UPLOAD_FAILURE"
    assert classify_root_cause(True, "PASS", "PASS", "FAIL", "FAIL") == "FFMPEG_SYNTHETIC_FAILURE"
    
    # All pass
    assert classify_root_cause(True, "PASS", "PASS", "PASS", "PASS") == "BOTH_TOOLS_PASS_UPLOAD"
    
    # Inconclusive (e.g. synth passes, upload is skipped/inconclusive, etc)
    assert classify_root_cause(True, "PASS", "PASS", "PASS", "SKIPPED") == "INCONCLUSIVE"
