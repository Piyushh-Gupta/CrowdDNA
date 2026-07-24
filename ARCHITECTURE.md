# CrowdDNA Architecture

CrowdDNA follows a modular, pipe-and-filter architecture separating data generation, training, and deployment.

## Simulation & Data Generation
Synthetic Crowd → ScenarioFactory → SimulationRunner → AutoLabeler → DatasetSerializer

## Model Architecture
- **GraphBuilder**: Converts spatial trajectories into PyTorch Geometric temporal graphs.
- **CrowdDNAGAT**: Spatial GNN that embeds pedestrian interactions.
- **TemporalEncoder**: GRU that captures temporal dynamics.
- **CrowdDNAModel**: High-level model integrating GAT, Temporal Encoder, and MLP classifier.

## Deployment Stack
- **ModelExporter**: Compiles CrowdDNAModel into TorchScript (`deployment.pt`).
- **InferenceRuntime**: Backend-agnostic abstraction for loading and executing compiled deployment artifacts (TorchScriptBackend, ONNXBackend).
- **SequenceBuffer**: Sliding-window buffer that assembles continuous temporal graph sequences into batch tensors for inference.

### Deployment Model Loading
`best.pt` (Training weights & config) → `training.validate_deployment` → `deployment.pt` (Stripped TorchScript binary) → `InferenceRuntime`

## Real-time Pipeline
**Pipeline Orchestrator (`CrowdFlowPipeline`) owns the complete lifecycle:**
Video Ingestion → YOLO Detection → ByteTrack Tracking → GraphBuilder → SequenceBuffer → InferenceRuntime → Annotation → App Dashboard (Gradio UI configured via `CROWDDNA_MODEL_PATH`)

---

## Principles

- Single Responsibility
- Deterministic Processing
- Modular Design
- Stable Interfaces
- Minimal Coupling