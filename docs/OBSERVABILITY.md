# Monitoring & Observability Framework

CrowdDNA includes a native, lightweight Monitoring & Observability subsystem. It operates silently to provide full visibility into training, evaluation, inference, and deployment environments.

## Running Observability

To launch observability explicitly:

```bash
python -m training.run_observability --action monitor --session-id "prod_run_001"
```

To run rapid health checks:

```bash
python -m training.run_observability --action health
```

To run a profiling session:

```bash
python -m training.run_observability --action profile
```

## Generated Artifacts
When a monitoring session completes, the framework exports data to `experiments/observability_exports/`:
- `session_*.json`: Full raw observation trace.
- `session_*_metrics.csv`: Flattened metric timeseries.
- `session_*_report.md`: Summary of events, health, and alerts.
- `*.png`: Configured matplotlib plots for latency, memory, throughput, and CPU.
