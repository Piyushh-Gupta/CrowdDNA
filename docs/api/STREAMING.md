# Streaming Architecture

Given that `CrowdFlowPipeline` processes video frames sequentially and produces a timeline, a streaming architecture is beneficial to provide live feedback to the Frontend Dashboard.

## Evaluated Options
1. **Polling (`GET /jobs/{id}`)**: Simple, but introduces latency and unnecessary server load.
2. **Server-Sent Events (SSE)**: Unidirectional streaming. Excellent for sending timeline updates as they are generated.
3. **WebSockets**: Bi-directional. Overkill for this use case since the client only receives data after the initial upload.

## Recommendation
**Server-Sent Events (SSE)** is the recommended architecture for the `/jobs/{id}/stream` endpoint. It perfectly matches the unidirectional flow of `PipelineResult.timeline` and is natively supported by modern browsers without the overhead of WebSockets.

*Note: This is an architectural recommendation only. No implementation is done in this iteration.*
