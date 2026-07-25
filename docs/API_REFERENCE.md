# API Reference

The CrowdDNA API (Phase 22) is a strict REST and Server-Sent Events (SSE) interface.

## Endpoints

### 1. Workflows
- `POST /api/v1/workflows` - Create a new workflow (Idempotent: requires `Idempotency-Key` header).
- `GET /api/v1/workflows/{id}` - Retrieve workflow status.

### 2. Inference
- `POST /api/v1/inference` - Trigger inference.
- `GET /api/v1/inference/{id}/stream` - SSE endpoint for streaming inference progress.

### 3. Operations & Health
- `GET /api/v1/health` - Liveness probe.
- `GET /api/v1/health/ready` - Readiness probe.

## Authentication & Authorization
All secured endpoints require a JWT token passed in the `Authorization: Bearer <token>` header. Role-based access control (RBAC) is enforced at the middleware layer.

## Pagination
List endpoints implement cursor-based pagination. Responses contain `next_cursor` within the `meta` envelope.

## Idempotency
All mutating requests (`POST`, `PUT`, `PATCH`, `DELETE`) require an `Idempotency-Key` header to prevent duplicate execution during network retries.

## Response Envelopes
All responses conform to a strict schema:
```json
{
  "data": { ... },
  "meta": {
    "request_id": "req-1234",
    "timestamp": "2026-07-25T00:00:00Z"
  }
}
```

## Error Model
Errors strictly follow RFC 7807 Problem Details:
```json
{
  "type": "https://errors.crowddna.internal/rate-limit",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "Too many requests. Please retry after 30 seconds."
}
```

## SDK Mapping
For language-specific integrations, refer to the [SDK Documentation](../sdk/docs/SDK.md).
