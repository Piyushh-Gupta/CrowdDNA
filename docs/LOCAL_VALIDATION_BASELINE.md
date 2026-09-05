# Local Validation Baseline

## 1. Purpose
This document establishes a known-good, verified local engineering baseline for the CrowdDNA repository. It records the successful end-to-end execution of the application on a local Windows machine, including GPU inference, FFmpeg streaming, the `CrowdFlowPipeline`, and the Gradio UI. 

This baseline serves as a reference point for future development and deployment debugging. **Cloud deployment is NOT validated by this baseline.**

## 2. Validation Date
**September 5, 2026**

## 3. Hardware / Environment
- **OS**: Windows
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (Architecture: Blackwell / sm_120)
- **NVIDIA Driver**: 592.27
- **Python**: 3.11.0
- **Isolated Environment**: `.venv-gpu` (Local validation only; production `requirements.txt` was unmodified)

## 4. GPU / PyTorch Compatibility
- **PyTorch**: 2.11.0+cu128
- **CUDA Runtime**: 12.8
- `torch.cuda.is_available()`: `True`
- **Verification**: Basic and advanced CUDA tensor matrix multiplications executed and passed successfully.

## 5. YOLO Validation
- **Ultralytics**: 8.1.29
- **YOLO Model**: `yolov8n.pt`
- **Device**: `cuda:0`
- **Verification**: 
  - Standalone YOLO inference executed successfully.
  - First-frame inference on the exact validation video completed successfully.
  - Detected objects in first frame: 2
  - CUDA synchronization confirmed.

## 6. Exact Input Video Fingerprint
- **Path**: `C:\Users\piyus\Downloads\15546948_1080_1920_50fps.mp4`
- **Size**: 21,308,462 bytes (~21.3 MB)
- **Resolution**: 1080x1920
- **FPS**: 50
- **Duration**: 11.04 seconds
- **Frames**: 552
- **Codec**: H.264
- **SHA-256**: `EF5022356EBD749D3066684145BF79F9D7F315567FBFC22F885481A18DA3EEF6`

*(Note: This is distinct from the smaller repository test fixture `tests/data/15546948_1080_1920_50fps.mp4`)*

## 7. FFmpeg Validation
- **FFmpeg executable**: `C:\Users\piyus\Downloads\ffmpeg_extracted\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.EXE`
- **FFprobe executable**: `C:\Users\piyus\Downloads\ffmpeg_extracted\ffmpeg-master-latest-win64-gpl\bin\ffprobe.EXE`
- **Discovery**: PASSED
- **Metadata Extraction**: PASSED
- **Streaming Frame Ingestion**: PASSED
- **First Frame Latency**: ~0.23 seconds
- **Sample Rate**: 5
- **Sampled Frames Yielded**: 111
- **Ingestion Time**: ~26.7 seconds
- **Subprocess Cleanup**: PASSED

## 8. Direct Pipeline Validation
Direct execution of `CrowdFlowPipeline.run()` against the validation video:
- **Metadata extraction**: PASSED
- **FFmpeg ingestion**: PASSED
- **YOLO execution**: PASSED
- **CUDA inference**: PASSED
- **ByteTracker**: PASSED
- **Graph construction**: PASSED
- **Risk inference**: Dummy mode
- **Output encoding**: PASSED
- **PipelineResult structure**: PASSED
- **Total processing time**: ~29.83 seconds

**Output Video**:
- **Exists**: YES
- **Size**: ~10.88 MB
- **Resolution**: 1080x1920
- **FPS**: 10
- **Duration**: ~11.1 seconds
- **Codec**: H.264
- **FFmpeg Decode Validation**: PASSED

## 9. Gradio Localhost Validation
The Gradio web application was tested end-to-end programmatically to verify UI API behavior:
- **Gradio Version**: 4.21.0
- **App Startup (`app.py`)**: PASSED
- **Startup Time**: ~3 seconds
- **Local URL**: `http://127.0.0.1:7860`
- **HTTP Availability**: PASSED
- **Video Upload**: PASSED (Validated via `/upload` multipart API)
- **Gradio Callback Invocation**: PASSED
- **CrowdFlowPipeline Invocation**: PASSED
- **YOLO CUDA Execution**: PASSED
- **Dummy Risk Mode**: PASSED
- **Output Generation**: PASSED
- **Output Returned to Gradio**: PASSED
- **Server Stability**: PASSED (Remained alive and responsive to subsequent HTTP requests)

*Note: Browser interaction was validated programmatically through the actual Gradio API path, simulating the exact UI payload rather than manually/visually clicking through a browser.*

## 10. Resource Measurements
Measurements observed locally during processing. **These are observed metrics, not hard limits or performance requirements.**
- **Baseline RSS (after module imports)**: ~686.85 MB
- **Peak RSS (during pipeline)**: ~1465 MB
- **Final RSS (post-processing)**: ~1465 MB
- **Peak GPU VRAM**: ~70.46 MB
- **CUDA Execution**: Confirmed actively utilizing the GPU

## 11. Streaming Architecture Confirmation
The current validated implementation correctly adheres to O(1) memory principles:
- Streams FFmpeg frames incrementally via subprocess pipes.
- Does **not** accumulate raw frames in memory.
- Does **not** accumulate annotated frames in memory.
- Does **not** invoke `list(frames_iter)`.
- Does **not** use the legacy OpenCV `VideoCapture` for production ingestion.
- Writes encoded output incrementally.
- The `PipelineResult` safely returns the `output_video_path` instead of retaining the heavy `annotated_frames` list.

## 12. Risk Model Limitation
At the time of validation, the `CROWDDNA_MODEL_PATH` environment variable was **unset**. 
Therefore, the pipeline executed successfully in **Dummy Mode**.
This confirms the orchestration, but does **not** imply that the production-trained CrowdDNA graph risk model has been validated. YOLO/vision validation and risk-model validation are explicitly separate concerns.

## 13. Known Limitations
- The Gradio 4.21.0 validation was performed via direct programmatic API calls because the local programmatic client environment mismatched Pydantic validation rules.
- Peak memory usage sits around 1.4 GB; this is perfectly acceptable for the localhost testing tier.
- Cloud deployment (e.g., Render/Vercel) remains entirely unvalidated by this local run.

## 14. Reproduction Instructions
To reproduce this exact validated state on the original Windows machine:

### Activate GPU environment
```powershell
.\.venv-gpu\Scripts\Activate.ps1
```

### Configure FFmpeg for the session
```powershell
$env:PATH = "C:\Users\piyus\Downloads\ffmpeg_extracted\ffmpeg-master-latest-win64-gpl\bin;" + $env:PATH
```

### Verify CUDA
```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Start application
```powershell
python app.py
```

### Access Local UI
```text
http://127.0.0.1:7860
```

## 15. Baseline Verdict
**LOCALHOST FULL WORKFLOW PASSED — CrowdDNA is locally validated end-to-end.**
