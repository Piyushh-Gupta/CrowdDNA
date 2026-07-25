# CrowdDNA API Guide

Welcome to the API & Service Layer for CrowdDNA.

## Versioning
All REST endpoints are versioned (e.g., `/api/v1/`). This guarantees backward compatibility as ML models evolve.

## Base URL
`http://localhost:8000/api/v1/`

## Authentication
Every endpoint (except `/health`) requires an `Authorization` header containing a valid identity token.

## Standard Response Format
All responses use a universal `ApiResponse` envelope.

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "errors": [],
  "metadata": {},
  "request_id": "uuid-1234",
  "correlation_id": "uuid-5678",
  "timestamp": 1690000000.0
}
```

### Error Response
```json
{
  "success": false,
  "data": null,
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Invalid parameters",
      "details": {}
    }
  ],
  "metadata": {},
  "request_id": "uuid-1234",
  "correlation_id": "uuid-5678",
  "timestamp": 1690000000.0
}
```

## Available Endpoints
- `POST /workflow`: Submit a workflow for asynchronous execution. Returns a Job ID.
- `GET /jobs`: Retrieve job status and history.
- `POST /inference`: Run inference on provided data.
- `GET /explainability`: Retrieve SHAP values and model explanations.
- `GET /observability`: Stream or retrieve telemetry and events.
- `GET /reproducibility`: Retrieve deterministic execution manifests.
- `GET /experiments`: List historical experiment metadata.
- `GET /security`: Access identity or role mappings.
- `GET /health`: Returns `{ startup, readiness, liveness }` tri-state health.

## Streaming
Streaming telemetry and inference progress can be accessed via WebSocket connections or Server-Sent Events (SSE). Documentation on streaming protocols is available in the OpenAPI specification.
