# Dependency Compatibility Matrix

This document outlines the strict deployment-critical dependency versions verified for production on Render. Modifying these without rigorous local validation may result in unexpected memory spikes, broken tensor operations, or UI failures.

| Package | Pinned Version | Purpose | Compatibility Notes | Upgrade Considerations |
|---------|---------------|---------|---------------------|------------------------|
| **Python** | `3.10` | Core Runtime | Python 3.10 provides optimal balance of PyTorch and ONNX compatibility. | Upgrading to 3.11/3.12 requires re-validating `torch-geometric` bindings. |
| **Docker Base** | `python:3.10-slim` | Container OS | `slim` avoids the 1GB overhead of the standard image, crucial for Render limits. | Requires manual installation of `libgl1-mesa-glx` for OpenCV. |
| **FastAPI** | *Pending (Iter 1B)* | API Layer | Will act as the core HTTP router for the inference engine. | Must align with Pydantic v2 compatibility. |
| **Gradio** | `4.21.0` | Frontend UI (Phase 8) | Provides the current web interface and video upload mechanisms. | Gradio 4.x introduces breaking changes from 3.x; bound strictly to `<4.24` for stability. |
| **gradio_client** | *Bundled* | UI Client | Required internally by Gradio for websocket/API communication. | Inherits Gradio's pinning. |
| **Pydantic** | *Pending (Iter 1B)* | Schema Validation | Will define the strict input/output DTOs for the API. | Must use v2.x for performance improvements. |
| **PyTorch** | `>=2.0.0` | ML Engine Core | Tensor operations and base for PyTorch Geometric. | Minor version upgrades are usually safe; test CPU inference latency. |
| **ultralytics** | `8.1.29` | Detection | YOLOv8 object detection wrapper. | Highly unstable API across minor versions. Do not upgrade without extensive testing of `tracker.py`. |
| **OpenCV** | `4.9.0.80` | Image Processing | Used for video decoding, frame extraction, and annotation drawing. | `opencv-python-headless` is strictly required to prevent X11 dependencies from crashing Docker. |

## Important Notes
- **PyTorch Installation**: PyTorch is installed implicitly via `torch-geometric` and `ultralytics`. The Dockerfile relies on the default PyPi wheels which include CPU-only binaries for Linux, ensuring the image remains as small as possible.
- **HuggingFace Hub**: Gradio `4.21.0` requires `huggingface-hub<0.24.0` to prevent an upstream breaking change.
