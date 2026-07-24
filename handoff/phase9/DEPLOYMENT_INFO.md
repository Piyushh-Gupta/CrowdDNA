# Deployment Artifact Lifecycle

This document provides context on the canonical deployment artifact used in Phase 9.

## Overview
The `deployment.pt` file is a TorchScript-compiled, stripped binary representation of the CrowdDNA model optimized for inference.

- **Source:** Generated deterministically from the validated training baseline (`experiments/runs/baseline_20260722_230537/checkpoints/best.pt`).
- **Version Control Policy:** Model binaries (including `.pt` artifacts) are **NOT** committed to Git to prevent repository bloat. They are distributed externally (e.g., via GitHub Releases or Google Drive).

## Reproducibility
If you need to regenerate the deployment artifact locally, use the official validation script:

```bash
python -m training.validate_deployment --checkpoint experiments/runs/baseline_20260722_230537/checkpoints/best.pt
```

### Expected Directory Layout
The official validation script expects and produces the following directory structure:

```
experiments/
└── runs/
    └── <experiment>/
        ├── checkpoints/
        │   └── best.pt           # Input training checkpoint
        └── deploy/
            ├── deployment.pt     # Generated output artifact
            └── report.md         # Deployment validation report
```
