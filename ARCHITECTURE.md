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
- **ModelExporter**: Compiles CrowdDNAModel into TorchScript and ONNX.
- **InferenceRuntime**: Backend-agnostic abstraction for loading and executing compiled models (TorchScriptBackend, ONNXBackend).
- **SequenceBuffer**: Sliding-window buffer that maintains continuous temporal sequences for real-time video processing.

## Real-time Pipeline
Video Ingestion → YOLO Detection → ByteTrack Tracking → GraphBuilder → SequenceBuffer → InferenceRuntime → App Dashboard

---

## Principles

- Single Responsibility
- Deterministic Processing
- Modular Design
- Stable Interfaces
- Minimal Coupling