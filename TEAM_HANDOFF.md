# CrowdDNA Team Handoff

## Project

CrowdDNA

An AI-powered crowd risk prediction system using synthetic crowd simulation, graph neural networks, temporal modelling and computer vision.

---

# Current Team

## Piyush Gupta (ML Subsystem)

Primary Responsibilities
- Model architecture (GAT, Temporal Encoder)
- Synthetic data generation & training pipelines
- Canonical checkpoints (`best.pt`)
- Deployment export (`validate_deployment`, `deployment.pt`)
- Inference runtime (`InferenceRuntime`)
- Sequence buffering (`SequenceBuffer`)
- CI/CD, Git workflow, repository architecture

## Aayushi (Application Layer)

Primary Responsibilities
- Application UI (Gradio `app.py`)
- Video rendering and annotation (`FrameAnnotator`, `TimelineBuilder`)
- End-to-end orchestration (`CrowdFlowPipeline`)
- User interaction and environment configuration (`CROWDDNA_MODEL_PATH`)

*Boundary Agreement:* The ML Subsystem provides generated `deployment.pt` artifacts and the `InferenceRuntime` API. The Application Layer consumes these artifacts via the `CrowdFlowPipeline` orchestration without modifying runtime internals.

---

# Development Philosophy

- Small feature branches
- One responsibility per module
- Review before merge
- Deterministic implementations
- Clean Git history