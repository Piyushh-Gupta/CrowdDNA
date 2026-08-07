# Render Deployment Setup

This document provides a comprehensive guide to deploying the CrowdDNA Gradio application on Render.

## Configuration
- **Platform**: Render Web Service
- **Runtime**: Docker
- **Repository Branch**: `develop` (or feature branch for testing)

## Environment Variables
### Required
- `PYTHONUNBUFFERED=1`: Ensures logs are streamed in real-time to the Render console.

### Optional
- `CROWDDNA_MODEL_PATH`: URL or file path to a `.pt` or `.onnx` model file. If left blank, the application operates in dummy mode.

## Hardware & Memory Recommendations
- **Minimum**: 2GB RAM (Standard Plan).
- **Recommended**: 4GB+ RAM. 
- *Note*: PyTorch, YOLOv8, and Gradio incur a significant memory overhead. Deploying on Render's Free tier (512MB RAM) will likely result in OOM (Out Of Memory) kills during initialization.

## Deployment Steps
1. Navigate to the Render Dashboard and click **New > Web Service**.
2. Connect your GitHub repository (`Piyushh-Gupta/CrowdDNA`).
3. Select the deployment branch.
4. Set the **Runtime** to `Docker`.
5. Clear the **Start Command** (Render will use the Dockerfile's `CMD`).
6. Add the required Environment Variables.
7. Click **Create Web Service**.

## Health Check Expectations
- Render automatically detects the exposed port (`$PORT`) and sends TCP health checks. 
- The deployment is considered healthy when the Gradio web server successfully binds to `0.0.0.0:$PORT` and accepts connections.

## Common Deployment Failures & Troubleshooting
- **OOMKilled (Exit Code 137)**: The container exceeded its memory limit. Upgrade the Render instance plan.
- **Port Binding Error**: Ensure `app.py` binds to `0.0.0.0` and `int(os.getenv("PORT", 7860))`. 
- **Missing Dependencies**: Check `requirements.txt` and ensure the Dockerfile successfully completed `pip install`. 

## Expected Startup Log Sequence
1. Docker image pull and extraction.
2. `python app.py` execution.
3. Gradio initialization.
4. "Running on local URL:  http://0.0.0.0:$PORT"

## Production Smoke Test Checklist
- [ ] Application loads and renders the UI without 500 errors.
- [ ] An uploaded MP4/AVI video is successfully processed.
- [ ] The annotated video output can be played in the browser.
- [ ] The risk timeline data renders correctly.
