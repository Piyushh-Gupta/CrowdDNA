# Observability Architecture

The Observability framework is designed for production deep learning environments, operating silently with zero blocking overhead to ensure `deployment.pt` integrity.

## Key Concepts

- **Immutable Observations**: `Observation`, `MetricObservation`, `EventObservation`, `TraceObservation`, `HealthObservation`, `AlertObservation`.
- **Event Bus & Tracing Engine**: Lock-free asynchronous buffer mechanisms.
- **Bounded Buffers**: `collections.deque(maxlen=N)` enforces strict memory ceilings preventing OOM failures during continuous video stream processing.
- **Pluggable Architecture**: Modules register dynamically via `ObservabilityRegistry`.

## Four-Stage Export Pipeline
1. **Collection**: Distributed hooks and periodic hardware polling emit observations subject to configured Sampling Policies (`Always`, `Adaptive`).
2. **Buffer**: Data is funneled into memory-bound lock-free deques.
3. **Aggregation**: The `AlertEngine` and `AggregationEngine` evaluate raw data for anomalies.
4. **Export**: Exporters cleanly serialize data into JSON, CSV, Markdown, and Matplotlib artifacts.
