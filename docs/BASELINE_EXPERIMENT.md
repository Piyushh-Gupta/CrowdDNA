# CrowdDNA Baseline Experiment Workflow

## Purpose
This document defines the canonical workflow for establishing the ground-truth baseline performance of the `CrowdDNAModel` against the synthetic `SequenceGraphDataset`. It provides the operational steps for generating a reproducible benchmark, validating the integrity of the end-to-end ML pipeline.

**Note**: This document describes the *workflow*, not the published benchmark results.

## How to Execute
Ensure that you have generated a synthetic dataset in `data/simulated` before proceeding.

To trigger the baseline workflow:
```bash
python -m training.run_baseline
```

## Expected Outputs
The script will load `experiments/baseline.yaml` and initialize an isolated runner instance. 
A typical console summary will read:
```
------------------------------------
CrowdDNA Baseline Workflow Complete
------------------------------------

Best Checkpoint: /path/to/experiments/runs/baseline_TIMESTAMP/checkpoints/best.pt
Training Time: ...
Evaluation Time: ...

Accuracy: ...
Precision: ...
Recall: ...
F1: ...

Experiment Directory: /path/to/experiments/runs/baseline_TIMESTAMP
------------------------------------
```

## Artifact Directory Layout
All generated assets will be stored cleanly in a timestamped folder located in `experiments/runs/`.
```
experiments/runs/baseline_YYYYMMDD_HHMMSS/
├── checkpoints/          # best.pt and latest.pt
├── config/               # snapshot.yaml
├── logs/                 # Console and execution traces
├── metadata/             # experiment_metadata.json
├── metrics/              # training_history.json, evaluation_metrics.json
└── reports/              # Visualizations (ROC, PR, CM) and report.md
```

## Future Comparisons
All subsequent experiments investigating architectural modifications, hyperparameters, or new data loaders must be executed using this exact pattern. Future branches should diff their `evaluation_metrics.json` against the assets produced by this baseline script to quantify performance drifts.
