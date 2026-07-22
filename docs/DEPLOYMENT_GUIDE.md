# CrowdDNA Deployment Guide

This guide details the process of converting a trained CrowdDNA model checkpoint into a production-ready deployment artifact.

## Export Process

The deployment artifact is generated deterministically from the canonical training checkpoint using the official validation script.

### Official Workflow

To generate the deployment artifact, run the validation tool on your best checkpoint:

```bash
python -m training.validate_deployment \
    --checkpoint experiments/runs/<experiment>/checkpoints/best.pt
```

### Required Inputs
- A fully trained `best.pt` checkpoint located in an experiment directory. This checkpoint contains both the model weights and the original architecture configuration used during training.

### Generated Outputs
The validation tool performs a strict architectural verification and a TorchScript compilation. If successful, it produces:
1. `deployment.pt`: The TorchScript-compiled model artifact, stripped of training-only parameters and optimized for inference.
2. `deployment_report.md`: A summary of the architectural validation and numerical equivalence checks.

### Directory Layout
The generated artifacts are written to the experiment's `deploy` directory:
```
experiments/
└── runs/
    └── <experiment>/
        ├── checkpoints/
        │   └── best.pt           # Input
        └── deploy/
            ├── deployment.pt     # Output artifact
            └── report.md
```

## Repository Policy

### Artifact Versioning
- **Do not commit `deployment.pt` to Git.** The repository does not version model binaries.
- The `best.pt` checkpoint is considered the canonical source of truth for the model state.
- `deployment.pt` is a dynamically generated deployment artifact.

### Integration
- Integration tests and production environments should use generated `deployment.pt` artifacts.
- Users must generate the deployment artifact locally using the official workflow or download it from an official release if one is published.
