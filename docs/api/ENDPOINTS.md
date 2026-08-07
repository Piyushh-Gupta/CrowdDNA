# API Endpoints

## System Endpoints
- `GET /health`: Returns overall system health.
- `GET /health/live`: Liveness probe for orchestration (e.g. Kubernetes). Returns 200 OK.
- `GET /health/ready`: Readiness probe to ensure models are loaded and GPU is accessible.
- `GET /version`: Returns the current API and Core Engine versions.

## Inference Endpoints
- `POST /inference`: Submit a video for crowd risk analysis. Uses `multipart/form-data`. Initiates an asynchronous job.
- `GET /jobs/{job_id}`: Poll for the status of an ongoing inference job. Returns `JobStatus`.
- `DELETE /jobs/{job_id}`: Cancel a running inference job.

## Model Management Endpoints
- `GET /models`: List currently available and loaded deployment models.
- `POST /models/load`: Dynamically swap the active model in the `InferenceRuntime` by specifying a model path/ID.

## Metrics Endpoints
- `GET /metrics`: Returns Prometheus-compatible system metrics (memory, latency, throughput).
