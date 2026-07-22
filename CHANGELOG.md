# Changelog

All notable changes to CrowdDNA will be documented here.

The format follows Keep a Changelog.

---

## [Unreleased]

### Added
- Phase 12C: Repository Hardening
- Open source readiness (LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)

### Changed
- Exception hierarchy decoupled from Exception, now inherits from CrowdFlowError
- Baseline script now parses configs safely with `from_dict`

---

## Phase 12 - Pipeline Integration

### Added
- SequenceBuffer for online inference
- Real-time Gradio UI integration in app.py

---

## Phase 11 - Inference Runtime & Export

### Added
- ONNX and TorchScript export
- Backend-agnostic InferenceRuntime
- InferenceResult contract

---

## Phase 8, 9, 10 - Training & Evaluation

### Added
- Training loop, Evaluation Engine
- Checkpoint manager and Early Stopping
- ExperimentRunner

---

## Phase 6, 7 - Architecture

### Added
- GAT (Graph Attention Network)
- Temporal Encoder (GRU)
- GraphBuilder, GraphDataset, SequenceDataset

---

## Phase 5 - Simulation Data

### Added
- Repository structure
- TrajectoryRecord, ScenarioFactory, SimulationRunner, AutoLabeler
- DatasetSerializer, generate_dataset()