# Authentication Strategy

Authentication is not currently implemented in the core engine. This outlines the future strategy for Iteration 4.

## Supported Methods
1. **API Keys**: For server-to-server integrations (SDK clients). Passed via the `X-API-Key` header.
2. **Bearer JWT**: For frontend dashboard users. Passed via the `Authorization: Bearer <token>` header.

## Future RBAC
Role-Based Access Control will differentiate between `Viewer` (read-only metrics/jobs) and `Admin` (can load models, run inference).
