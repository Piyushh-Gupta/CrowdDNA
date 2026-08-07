# Error Model

The API adopts the RFC 7807 Problem Details for HTTP APIs standard.

## Structure
All error responses return `application/problem+json`:

```json
{
  "type": "https://api.crowddna.internal/errors/invalid-video",
  "title": "Invalid Video Format",
  "status": 400,
  "detail": "The uploaded file is not a supported MP4 or AVI format.",
  "instance": "/inference"
}
```

## Standard Status Codes
- `200 OK`: Successful GET or completed synchronous POST.
- `201 Created`: Job successfully created.
- `202 Accepted`: Asynchronous job queued.
- `204 No Content`: Successful DELETE.
- `400 Bad Request`: Validation errors (e.g. unsupported video format).
- `401 Unauthorized`: Missing or invalid credentials.
- `403 Forbidden`: Insufficient permissions.
- `404 Not Found`: Resource (Job, Model) not found.
- `409 Conflict`: Attempting to process conflicting states.
- `422 Unprocessable Entity`: Schema validation failure.
- `429 Too Many Requests`: Rate limit exceeded.
- `500 Internal Server Error`: `ModelInferenceError` or `CrowdFlowError` uncaught exceptions.
