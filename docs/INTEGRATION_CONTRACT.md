# CrowdDNA — Deployment / Inference Integration Contract

> **Version:** 1.0  
> **Status:** Authoritative  
> **Date:** 2026-07-22  
> **Scope:** Everything that crosses the boundary between the Phase 11 Deployment Subsystem (Piyush) and the Phase 7/8/9 Inference Pipeline (Aayushi).

---

## Table of Contents

1. [Component Ownership](#1-component-ownership)
2. [End-to-End Deployment Flow](#2-end-to-end-deployment-flow)
3. [Runtime API Contract](#3-runtime-api-contract)
4. [Tensor Input Contracts](#4-tensor-input-contracts)
5. [GraphBuilder Output Contract](#5-graphbuilder-output-contract)
6. [Sequence Buffering Strategy](#6-sequence-buffering-strategy)
7. [InferenceResult Contract](#7-inferenceresult-contract)
8. [Runtime Error Handling](#8-runtime-error-handling)
9. [Performance Expectations and Latency Budget](#9-performance-expectations-and-latency-budget)
10. [Future Extension Points](#10-future-extension-points)
11. [Architecture Rationale](#11-architecture-rationale)
12. [Assumptions](#12-assumptions)
13. [Integration Risks](#13-integration-risks)
14. [Recommended Implementation Order](#14-recommended-implementation-order)

---

## 1. Component Ownership

### ML Subsystem (Piyush)

| Module | Path | Responsibility |
|---|---|---|
| `CrowdDNADeploymentModel` | `crowdflow_dna/model/deployment_model.py` | Pure-tensor forward pass for inference. Owned entirely by ML subsystem. |
| `ModelExporter` / `ExportResult` | `training/export_model.py` | Serialises the trained model to TorchScript (`.pt`) and ONNX (`.onnx`). Produces `metadata.json`. Runs offline. |
| `InferenceRuntime` | `crowdflow_dna/inference/runtime.py` | Loads exported models, routes to `TorchScriptBackend` or `ONNXBackend`, times inference, returns `InferenceResult`. |
| `InferenceBackend` (ABC) | `crowdflow_dna/inference/runtime.py` | Pluggable backend interface. |
| `SequenceBuffer` | `crowdflow_dna/inference/sequence_buffer.py` | Sliding-window buffer that assembles continuous temporal graph sequences into batch tensors. |
| `GraphBuilder` | `crowdflow_dna/graph/graph_builder.py` | Converts `TrackItem` lists into per-frame `torch_geometric.data.Data` graphs. |
| `InferenceResult` | `crowdflow_dna/inference/runtime.py` | Canonical result dataclass. Shared across both teams — treat as immutable public API. |

### Application Layer (Aayushi)

| Module | Path | Responsibility |
|---|---|---|
| `CrowdFlowPipeline` | `crowdflow_dna/pipeline.py` | Orchestrates all stages end-to-end. Owns the inference lifecycle (`SequenceBuffer` & `InferenceRuntime`). |
| `VideoIngestor` | `crowdflow_dna/ingestion/video_loader.py` | Decodes video frames at `FRAME_SAMPLE_RATE`. |
| `Yolov8Detector` | `crowdflow_dna/detection/detector.py` | Detects pedestrians per frame; produces `Detection` objects. |
| `ByteTracker` | `crowdflow_dna/tracking/tracker.py` | Assigns track IDs and estimates velocity; produces `TrackItem` objects. |
| `FrameAnnotator` | `crowdflow_dna/rendering/video_renderer.py` | Draws bounding boxes and risk labels onto frames. Accepts `List[RiskPrediction]`. |
| `TimelineBuilder` | `crowdflow_dna/rendering/timeline.py` | Accumulates per-frame predictions into a `List[TimelineEntry]`. |
| `app.py` | `app.py` | Gradio UI; configures `CrowdFlowPipeline` via `CROWDDNA_MODEL_PATH` environment variable. |

### Shared Schemas (Neutral — neither team modifies without consensus)

| Schema | Path | Fields |
|---|---|---|
| `TrackItem` | `crowdflow_dna/schemas.py` | `track_id`, `bbox`, `centroid`, `velocity` |
| `RiskPrediction` | `crowdflow_dna/schemas.py` | `region_id`, `label`, `confidence` |
| `InferenceResult` | `crowdflow_dna/inference/runtime.py` | See §7 |

---

### Offline Deployment Flow

```
best.pt (Training Checkpoint)
        │
        ▼  [Export & Validation]
training.validate_deployment
        │
        ▼
deployment.pt (TorchScript Artifact)
        │
        ▼
InferenceRuntime.load_model()
```

### Real-Time Inference Flow

The complete data path from raw video to annotated output is:

```
Video file (MP4 / AVI)
        │
        ▼  [Stage 1 — Ingestion]
VideoIngestor.load(video_path)
  → List[np.ndarray]  (BGR frames, sampled every FRAME_SAMPLE_RATE=5 frames)
  → metadata dict     (fps, width, height, frame_count, duration_seconds, sample_rate)
        │
        ▼  [Stage 2 — Per-Frame: Detection]
Yolov8Detector.detect(frame)
  → List[Detection]   (bbox, confidence per detected person)
        │
        ▼  [Stage 3 — Per-Frame: Tracking]
ByteTracker.update(detections)
  → List[TrackItem]   (track_id, bbox, centroid, velocity)
        │
        ▼  [Stage 4 — Per-Frame: Graph Construction]
GraphBuilder.build(positions, velocities)
  → torch_geometric.data.Data
      x:          (N, 5) float32    # [x_norm, y_norm, vx, vy, speed]
      edge_index: (2, E) long       # COO proximity edges
      edge_attr:  (E, 4) float32    # [dx, dy, distance, relative_speed]
        │
        ▼  [Stage 5 — Sequence Buffering]
SequenceBuffer.push(graph)          # per-track-session window management
  → when buffer is full:
      x:          (total_nodes, 5)  # flat-concatenated across W frames
      edge_index: (2, total_edges)  # with per-frame node offsets applied
      edge_attr:  (total_edges, 4)
      batch:      (total_nodes,)    # node → frame index mapping
      seq_lengths:(B,)              # number of frames per trajectory
        │
        ▼  [Stage 6 — Inference]
InferenceRuntime.predict(x, edge_index, edge_attr, batch, seq_lengths)
  → InferenceResult
      predicted_class:   int            # 0=Safe, 1=Congesting, 2=Critical
      probabilities:     np.ndarray     # (num_classes,) softmax distribution
      confidence:        float          # max probability
      backend:           str            # "TorchScript" or "ONNX Runtime"
      inference_time_ms: float
      model_format:      str
      model_version:     str | None
        │
        ▼  [Stage 7 — Translation]
InferenceResult → RiskPrediction
      region_id:  = track_id (or 0 for scene-level)
      label:      = CLASS_NAMES[predicted_class]  # "Safe","Congesting","Critical"
      confidence: = InferenceResult.confidence
        │
        ▼  [Stage 8 — Annotation]
FrameAnnotator.annotate(frame, tracks, predictions)
  → np.ndarray (annotated BGR frame)
        │
        ▼  [Stage 9 — Timeline]
TimelineBuilder.record(frame_index, predictions)
  → List[TimelineEntry]
        │
        ▼  [Stage 10 — Gradio UI]
app.py / Gradio Blocks
  → annotated video stream
  → risk timeline table
  → metadata table
```

### Current Pipeline State

`CrowdFlowPipeline.__init__` accepts `model_path: str | Path | None` and fully owns the lifecycle of `SequenceBuffer` and `InferenceRuntime`. This improves encapsulation while preserving the external inference contract. 
`app.py` reads `CROWDDNA_MODEL_PATH` from the environment. If the variable exists and loading succeeds, it runs in **inference mode**. If it is missing or loading fails, it gracefully falls back to **dummy mode**.

---

## 3. Runtime API Contract

The `InferenceRuntime` class is the **only** interface the pipeline team may use. All other deployment classes (`CrowdDNADeploymentModel`, `ModelExporter`, `TorchScriptBackend`, `ONNXBackend`) are internal to the deployment subsystem.

### 3.1 `InferenceRuntime.__init__()`

```python
runtime = InferenceRuntime()
```

Creates an unloaded runtime. No arguments. State is `backend=None`.

### 3.2 `InferenceRuntime.load_model(path, version=None)`

```python
runtime.load_model(
    path: Path | str,       # absolute or relative path to .pt or .onnx file
    version: str | None,    # optional version tag; stored in InferenceResult
)
```

**Behaviour:**
- Determines backend from file extension: `.pt` → `TorchScriptBackend`; `.onnx` → `ONNXBackend`.
- Raises `ModelNotFoundError` if the path does not exist.
- Raises `UnsupportedModelFormatError` if the extension is neither `.pt` nor `.onnx`.
- Raises `InferenceExecutionError` if the model file is corrupt or incompatible.
- Must be called exactly once before any `predict` call.

**Thread safety:** Not guaranteed. Initialise one `InferenceRuntime` per worker process.

### 3.3 `InferenceRuntime.predict(x, edge_index, edge_attr, batch, seq_lengths)`

```python
result: InferenceResult = runtime.predict(
    x:           Tensor,   # (total_nodes, 5)    float32
    edge_index:  Tensor,   # (2, total_edges)    int64
    edge_attr:   Tensor,   # (total_edges, 4)    float32
    batch:       Tensor,   # (total_nodes,)      int64
    seq_lengths: Tensor,   # (B,)                int64
)
```

Returns the `InferenceResult` for the **first** trajectory in the batch (index `[0]`). See §4 for full tensor shape specification.

### 3.4 `InferenceRuntime.predict_batch(x, edge_index, edge_attr, batch, seq_lengths)`

```python
results: list[InferenceResult] = runtime.predict_batch(
    x, edge_index, edge_attr, batch, seq_lengths
)
```

Returns one `InferenceResult` per trajectory in the batch. `len(results) == seq_lengths.shape[0]`. All results share the same `inference_time_ms` (total batch time, not per-sample).

---

## 4. Tensor Input Contracts

These are the **exact** tensor shapes and dtypes `InferenceRuntime` expects. They must be constructed by the pipeline team's sequence buffering layer (§6).

### `x` — Node Feature Matrix

| Property | Value |
|---|---|
| Shape | `(total_nodes, 5)` |
| dtype | `torch.float32` |
| total_nodes | Sum of nodes across all frames across all sequences in the batch |
| Column 0 | Normalised x position ∈ `[0, 1]` (cx / frame_width) |
| Column 1 | Normalised y position ∈ `[0, 1]` (cy / frame_height) |
| Column 2 | x-velocity, pixel-per-frame (not normalised) |
| Column 3 | y-velocity, pixel-per-frame (not normalised) |
| Column 4 | Speed = `sqrt(vx² + vy²)`, always non-negative |

> **Important:** Node features are produced verbatim from `GraphBuilder.build()`. The `GraphBuilder` sets columns `[x, y, vx, vy, speed]` — this maps directly to the required feature vector. No additional transformation is needed.

### `edge_index` — COO Edge Indices

| Property | Value |
|---|---|
| Shape | `(2, total_edges)` |
| dtype | `torch.int64` |
| total_edges | Sum of directed edges across all frames in the batch |
| Row 0 | Source node indices (global, within the concatenated `x` tensor) |
| Row 1 | Destination node indices |
| Indexing | **Global** (0-based, relative to the full `x` tensor, not per-frame) |

> **Critical:** When concatenating graphs from multiple frames, node indices in `edge_index` must be **offset** by the cumulative node count of all previous frames. Failure to do this causes edge indices to point to wrong nodes.

### `edge_attr` — Edge Feature Matrix

| Property | Value |
|---|---|
| Shape | `(total_edges, 4)` |
| dtype | `torch.float32` |
| Column 0 | `dx` = dst_x − src_x (signed, pixel-scale) |
| Column 1 | `dy` = dst_y − src_y (signed, pixel-scale) |
| Column 2 | Euclidean distance between nodes (always ≥ 0) |
| Column 3 | Relative speed = `‖vel_dst − vel_src‖₂` |

> These are produced verbatim by `GraphBuilder._build_edges()`. No transformation needed.

### `batch` — Node-to-Frame Assignment

| Property | Value |
|---|---|
| Shape | `(total_nodes,)` |
| dtype | `torch.int64` |
| Values | Integer in `[0, total_frames_in_batch)` |
| Meaning | Maps each node to the frame-graph it belongs to |
| Construction | `batch[k] = frame_index` for each node `k` in that frame |

> The deployment model uses `batch` to group nodes by frame during GAT pooling. This is analogous to `torch_geometric.data.Batch.batch`.

### `seq_lengths` — Sequence Lengths

| Property | Value |
|---|---|
| Shape | `(B,)` |
| dtype | `torch.int64` |
| Values | Number of frames in each trajectory sequence in the batch |
| Constraint | `sum(seq_lengths) == total_frames_in_batch` |
| Constraint | Every entry must be ≥ 1 |

> **Example:** Two trajectories, one with 10 frames and one with 12 frames: `seq_lengths = torch.tensor([10, 12])`. Total frame-graphs in `batch` would span indices 0–21.

---

## 5. GraphBuilder Output Contract

`GraphBuilder.build(positions, velocities)` returns a `torch_geometric.data.Data` object. The pipeline team uses this as the source for tensor assembly.

### Inputs to `GraphBuilder.build()`

| Argument | Shape | dtype | Source |
|---|---|---|---|
| `positions` | `(N, 2)` | float32 | `[(t.centroid[0] / width, t.centroid[1] / height) for t in tracks]` |
| `velocities` | `(N, 2)` | float32 | `[(t.velocity[0], t.velocity[1]) for t in tracks]` |

`N` = number of active tracks in this frame. Both arrays must have the same `N`.

### Outputs from `GraphBuilder.build()`

| Attribute | Shape | dtype | Notes |
|---|---|---|---|
| `.x` | `(N, 5)` | float32 | `[x_norm, y_norm, vx, vy, speed]` — directly maps to runtime tensor `x` |
| `.edge_index` | `(2, E)` | int64 | **Local** indices (0 to N-1). Must be offset before concatenation. |
| `.edge_attr` | `(E, 4)` | float32 | `[dx, dy, distance, relative_speed]` — directly maps to runtime tensor `edge_attr` |
| `.num_nodes` | `int` | — | Equal to `N` |

### Empty-Frame Behaviour

When `N < 2`, `GraphBuilder` returns a valid `Data` object with:
- `.x` of shape `(N, 5)` (may be `(0, 5)` if no tracks)
- `.edge_index` of shape `(2, 0)` — no edges
- `.edge_attr` of shape `(0, 4)` — no edge features

The buffering layer (§6) must handle empty graphs. The deployment model's forward pass tolerates them.

---

## 6. Sequence Buffering Strategy

This section defines the protocol for converting per-frame `Data` objects (produced by `GraphBuilder`) into the batch of flat tensors consumed by `InferenceRuntime`.

### 6.1 Window Parameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| `WINDOW_SIZE` | 10 frames | Number of consecutive frames forming one inference request |
| `STRIDE` | 5 frames | Inference runs every 5 frames; window slides forward by 5 |
| `WARM_UP_FRAMES` | 10 | No inference until the buffer contains `WINDOW_SIZE` frames |

These are recommendations. The pipeline team owns the actual configuration values. The deployment subsystem has no dependency on these values.

### 6.2 Buffer Lifecycle

```
Frame 0:  buffer = [G0]                          — insufficient; no inference
Frame 1:  buffer = [G0, G1]                      — insufficient; no inference
...
Frame 9:  buffer = [G0, G1, ..., G9]            — full; inference fires
Frame 10: buffer = [G0, G1, ..., G10]           — slide; evict G0; inference fires
...
```

For each inference call, the buffer assembles a single-sequence batch (`B=1`, `seq_lengths=[W]`).

### 6.3 Tensor Assembly from Buffer

Given a buffer of `W` `Data` objects `[G0, G1, ..., G_{W-1}]`:

**Step 1 — Compute cumulative node offsets:**
```
offset[0] = 0
offset[i] = offset[i-1] + G_{i-1}.num_nodes
```

**Step 2 — Shift edge indices:**
```
G_i.edge_index_global = G_i.edge_index + offset[i]
```

**Step 3 — Concatenate tensors:**
```
x           = torch.cat([G_i.x for i in 0..W-1], dim=0)          # (total_nodes, 5)
edge_index  = torch.cat([G_i.edge_index_global for i in ...], dim=1)  # (2, total_edges)
edge_attr   = torch.cat([G_i.edge_attr for i in ...], dim=0)      # (total_edges, 4)
```

**Step 4 — Build batch vector:**
```
batch = torch.cat([
    torch.full((G_i.num_nodes,), i, dtype=torch.long)
    for i in 0..W-1
], dim=0)                                                          # (total_nodes,)
```

**Step 5 — Build seq_lengths:**
```
seq_lengths = torch.tensor([W], dtype=torch.long)                  # (1,)
```

### 6.4 Missing Detections

When a frame has zero tracks (`N=0`), the buffer must still include a placeholder entry:
- Insert a `Data` object with `.x = torch.zeros((0, 5))`, `.edge_index = torch.zeros((2, 0), dtype=torch.long)`, `.edge_attr = torch.zeros((0, 4))`, `.num_nodes = 0`.
- The node offset for this frame is 0 (no contribution).
- The `batch` vector receives no entries for this frame.
- `seq_lengths[i]` is still counted as one frame consumed.

### 6.5 Session Reset

The buffer must be **fully cleared** on:
- Video end (pipeline `reset()` or new `pipeline.run()` call)
- Camera switch (multi-camera scenarios)
- Track-ID discontinuity exceeding a configurable gap threshold

The pipeline team is responsible for detecting reset conditions and calling `buffer.clear()` before the next `push`.

---

## 7. InferenceResult Contract

`InferenceResult` is a **frozen dataclass**. It must be treated as read-only once returned. The pipeline team must not subclass it or add attributes to it.

```
InferenceResult
├── predicted_class: int
│     Integer index of the highest-probability class.
│     Canonical mapping:
│       0 → "Safe"
│       1 → "Congesting"
│       2 → "Critical"
│     Must be used with CLASS_NAMES = ["Safe", "Congesting", "Critical"]
│     defined by the pipeline integration layer — NOT inside the runtime.
│
├── probabilities: np.ndarray
│     Shape: (num_classes,)  — always 3 for the current model
│     dtype: float64 (NumPy default after softmax)
│     Guaranteed to be a proper probability distribution: sum = 1.0 ± 1e-6
│     Always np.ndarray regardless of backend (TorchScript or ONNX).
│
├── confidence: float
│     Scalar. Equal to max(probabilities). Range: (0.0, 1.0].
│     Directly usable as RiskPrediction.confidence.
│
├── backend: str
│     "TorchScript"   — when loaded from a .pt file
│     "ONNX Runtime"  — when loaded from an .onnx file
│
├── inference_time_ms: float
│     Wall-clock time from backend.predict() entry to softmax output.
│     Measured with time.perf_counter(). Does not include data preparation
│     by the calling layer. For predict_batch(), this is the total batch
│     time and is identical across all results in the returned list.
│
├── model_format: str
│     "TorchScript" or "ONNX". Matches backend for all current use cases.
│
└── model_version: str | None
      Populated from the `version` argument passed to load_model().
      None if not provided. Pipeline team may pass the git commit hash
      or release tag here for traceability.
```

### Translating `InferenceResult` to `RiskPrediction`

The pipeline integration layer is responsible for this translation. The deployment subsystem does not produce `RiskPrediction` objects.

```python
CLASS_NAMES = ["Safe", "Congesting", "Critical"]  # defined by pipeline team

def result_to_prediction(result: InferenceResult, track_id: int) -> RiskPrediction:
    return RiskPrediction(
        region_id=track_id,
        label=CLASS_NAMES[result.predicted_class],
        confidence=result.confidence,
    )
```

---

## 8. Runtime Error Handling

### Exception Hierarchy

```
InferenceExecutionError     — catch at pipeline level, log, skip frame
ModelNotFoundError          — catch at startup, fail fast with clear message
UnsupportedModelFormatError — catch at startup, fail fast with clear message
```

All three are importable from `crowdflow_dna.inference`:
```python
from crowdflow_dna.inference import (
    InferenceRuntime,
    ModelNotFoundError,
    UnsupportedModelFormatError,
    InferenceExecutionError,
)
```

### Error Handling Protocol

| Condition | Exception | Required Pipeline Behaviour |
|---|---|---|
| Model file not found at startup | `ModelNotFoundError` | Abort startup. Log path. Do not enter dummy mode silently. |
| Unknown file extension (not `.pt` or `.onnx`) | `UnsupportedModelFormatError` | Abort startup. Log the extension. |
| TorchScript model corrupt or incompatible | `InferenceExecutionError` (from `load`) | Abort startup. Log underlying cause. |
| ONNX Runtime not installed | `InferenceExecutionError` (from `load`) | Abort startup if ONNX backend requested. Log `pip install onnxruntime`. |
| Tensor shape mismatch during inference | `InferenceExecutionError` (from `predict`) | Catch at frame level. Skip inference for that frame. Log frame index and shapes. |
| Any other inference failure | `InferenceExecutionError` (from `predict`) | Catch at frame level. Skip frame. Do not propagate to Gradio. |
| `predict` called before `load_model` | `InferenceExecutionError` | Catch at pipeline init. This is a programming error — must not occur in production. |

### Integration with Pipeline Error Types

The pipeline already defines `ModelInferenceError` in `crowdflow_dna/errors.py`. The recommended pattern is:

```python
try:
    result = runtime.predict(x, edge_index, edge_attr, batch, seq_lengths)
except InferenceExecutionError as exc:
    raise ModelInferenceError(f"InferenceRuntime failed on frame {frame_idx}: {exc}") from exc
```

This preserves the pipeline's existing error taxonomy while surfacing deployment-specific failures.

---

## 9. Performance Expectations and Latency Budget

### Frame-Level Latency Budget

The pipeline samples every 5th frame (`FRAME_SAMPLE_RATE = 5`). At 30 fps source video, this gives one pipeline execution every ~167 ms. The total inference budget for Stage 5+6 (buffering + runtime) within that window is:

| Stage | Budget |
|---|---|
| Sequence buffer assembly (tensor cat + offset) | < 2 ms |
| InferenceRuntime.predict (TorchScript, CPU) | < 20 ms |
| InferenceRuntime.predict (ONNX Runtime, CPU) | < 30 ms |
| InferenceRuntime.predict (TorchScript, GPU) | < 5 ms |
| Total inference budget | **< 35 ms** |

> These are targets, not guarantees. Actual times depend on hardware and sequence length. `InferenceResult.inference_time_ms` provides empirical measurements.

### Throughput Expectations

- Single-sequence batch (`B=1`, `W=10`): Primary mode for Aayushi's video pipeline.
- Multi-sequence batch (`B>1`): Used for evaluation harness, not in live inference.

### Model Size Reference

| Export Artifact | Estimated Size |
|---|---|
| `deployment.pt` (TorchScript) | ~2–5 MB (depends on head size) |
| `crowddna.onnx` (ONNX) | ~2–5 MB |
| `metadata.json` | < 1 KB |

---

## 10. Future Extension Points

The `InferenceBackend` abstract class is the primary extension mechanism. Adding a new inference engine requires:
1. Subclass `InferenceBackend` implementing `load(path)` and `predict(...)`.
2. Register the new backend in `InferenceRuntime.load_model()` by file extension or a new `backend_type` argument.
3. No changes to pipeline code.

### Planned Extension Points

| Extension | Backend Class | Extension Point | Notes |
|---|---|---|---|
| TensorRT | `TensorRTBackend` | `.engine` extension | Requires NVIDIA GPU + TensorRT SDK. `load()` wraps `tensorrt.Runtime`. |
| OpenVINO | `OpenVINOBackend` | `.xml` / `.bin` extension | Intel CPU acceleration. Wraps `openvino.runtime.Core`. |
| Triton Inference Server | `TritonBackend` | URL or endpoint string | gRPC / HTTP client. Enables remote inference for multi-camera deployments. |
| Quantised INT8 | Handled inside existing backends | Same `.pt` or `.onnx` format | Quantised models are drop-in replacements. No API change. |
| Streaming inference | `StreamingBuffer` | Wraps `SequenceBuffer` | Emits partial results before buffer fills. Requires model to accept variable seq_lengths. |
| Multi-camera | Multiple `InferenceRuntime` instances | One per camera | Independent `load_model()` calls. No shared state. |

---

## 11. Architecture Rationale

### Why a flat-tensor API instead of `torch_geometric.data.Batch`?

`CrowdDNADeploymentModel` was deliberately designed to accept raw tensors (`x`, `edge_index`, `edge_attr`, `batch`, `seq_lengths`) rather than PyTorch Geometric `Data`/`Batch` objects. The reasons are:

1. **TorchScript incompatibility:** `torch.jit.script()` cannot trace through PyG's Python-object-heavy API. `Batch.from_data_list()` internally uses Python `list` comprehensions and dynamic dispatch that the TorchScript static compiler cannot handle.
2. **ONNX incompatibility:** PyG's `scatter_reduce` and `to_dense_batch` operations do not have stable ONNX opset mappings. Flat-tensor padding via `torch.where` and `torch.arange` is fully traceable.
3. **Deployment portability:** The flat-tensor API is independent of PyTorch Geometric versioning. A future engine (TensorRT, OpenVINO) only sees NumPy arrays and standard PyTorch tensors.

### Why does GraphBuilder still return `torch_geometric.data.Data`?

`GraphBuilder` was built before the deployment model existed and is used by the training pipeline which *does* use PyG natively. Changing its output format would break the training pipeline. The correct solution is the **buffering adapter layer** (§6) that the pipeline team implements, which converts `Data` objects into the flat tensors required by the runtime.

### Why does `CrowdFlowPipeline` directly encapsulate the inference lifecycle?

The pipeline directly accepts `model_path` and completely owns the instantiation and lifecycle of both `SequenceBuffer` and `InferenceRuntime`. This replaces the older `model_fn` callable injection pattern. This design provides superior encapsulation because it guarantees that buffer resets, error mapping (`ModelInferenceError`), and configuration (`CROWDDNA_MODEL_PATH`) are handled uniformly within the pipeline boundary rather than leaking into the Gradio UI layer (`app.py`).

## 12. Assumptions

1. **Single global risk class per inference call.** The current model (`CrowdDNADeploymentModel`) returns a single logit vector of shape `(batch_size, num_classes)`. It classifies the entire window of frames as one risk state, not per-track or per-region. The pipeline team should map this single `InferenceResult` to all active tracks in the frame using a uniform `region_id` scheme (e.g., `region_id=0` for the scene).

2. **Fixed node feature dimension of 5.** The model was trained with `in_channels=5` (`[x, y, vx, vy, speed]`). The `GraphBuilder` always produces `(N, 5)` node features. This must not change without retraining the model.

3. **Fixed edge feature dimension of 4.** The GAT was trained with `edge_dim=4` (`[dx, dy, distance, relative_speed]`). The `GraphBuilder` always produces `(E, 4)` edge features. This must not change without retraining.

4. **Fixed num_classes of 3.** The classifier head has 3 outputs: `Safe`, `Congesting`, `Critical`. `InferenceResult.probabilities` will always be of length 3.

5. **Sequence lengths are variable.** Although the training pipeline used fixed-length sequences of 300 timesteps, the deployment model handles variable `seq_lengths` via vectorised padding. The pipeline team may use shorter windows (e.g., 10 frames) without retraining.

6. **TorchScript is the primary backend.** ONNX export is best-effort due to PyG operator compatibility. The `.pt` file is the guaranteed deployment artifact. The pipeline team should default to `.pt` unless there is a specific ONNX requirement.

7. **No stateful buffers inside the runtime.** `InferenceRuntime` is stateless after `load_model()`. Sequence history management lives entirely in the pipeline team's buffering layer. Concurrent calls to `predict` from different threads are safe only if each thread has its own `InferenceRuntime` instance.

8. **Position normalisation is the pipeline team's responsibility.** `CrowdFlowPipeline._extract_arrays()` already normalises centroids by frame dimensions before passing to `GraphBuilder`. This must continue to be done correctly.

---

## 13. Integration Risks

| Risk | Severity | Owner | Mitigation |
|---|---|---|---|
| **Tensor contract violation** — pipeline passes wrong shapes to `predict()` | **High** | Pipeline team | Add shape-assertion guards in the buffering layer before calling `predict`. Log shapes on `InferenceExecutionError`. |
| **Edge index not offset** — concatenated graphs have wrong cross-frame edges | **High** | Pipeline team | Unit-test the assembly function against a known graph with manually computed expected edges. |
| **Warm-up frames produce no inference** — first WINDOW_SIZE frames have no risk output | **Medium** | Pipeline team | Document in Gradio UI status. Show "Warming up…" for the first N frames. |
| **Empty frames corrupt batch tensor** — zero-node frames not handled correctly | **Medium** | Pipeline team | Test with zero-track frames explicitly. The buffer must insert placeholder `Data` objects. |
| **ONNX backend unavailable** — `onnxruntime` not installed in deployment environment | **Medium** | Deployment team | Default to TorchScript. Document `onnxruntime` as optional in `requirements.txt`. |
| **Single-class result mapped to multiple tracks** — uniform label applied to all tracks | **Low** | Pipeline team | Acceptable for Phase 11. Scene-level classification is the current model design. |
| **Checkpoint state_dict key mismatch** — `ModelExporter` loads keys that don't match `CrowdDNADeploymentModel` | **Low** | Deployment team | Export tests already verify this. The exporter uses `deploy_model.load_state_dict(state_dict)` which will raise on mismatch. |
| **Version drift** — pipeline uses stale cached `.pt` file while model is updated | **Low** | Both teams | Always pass `version=git_commit_hash` to `load_model()`. Surface `model_version` in Gradio metadata table. |

---

## 14. Recommended Implementation Order

The following order minimises integration risk and allows independent development:

1. **[Pipeline team]** Implement a standalone `SequenceBuffer` class that accepts `Data` objects, manages a sliding window, and produces the 5-tensor tuple. Unit-test the assembly function against `GraphBuilder` output directly (no model needed).

2. **[Deployment team]** Export the trained model using `ModelExporter`. Verify `deployment.pt` loads correctly with `InferenceRuntime`. Commit the `.pt` file or document a download URL.

3. **[Pipeline team]** Write an integration smoke test that loads the `.pt` file, passes a synthetic 5-tensor tuple of the correct shapes, and asserts `InferenceResult.predicted_class in {0, 1, 2}`.

4. **[Pipeline team]** Wire `InferenceRuntime` into `CrowdFlowPipeline` directly. Replace dummy mode initialization in `app.py` with `CROWDDNA_MODEL_PATH` configuration.

5. **[Both teams]** End-to-end test on a short video clip. Measure `InferenceResult.inference_time_ms` over 30+ frames. Verify it is within the 35 ms budget.

6. **[Both teams]** Update `app.py` status message to surface `model_version` and backend name from `InferenceResult`.

7. **[Deployment team]** (Optional) Export ONNX artifact. Provide to pipeline team for testing with `ONNXBackend`.

8. **[Both teams]** Document any per-deployment configuration (model path, version tag) in `experiments/baseline.yaml` or a new `configs/inference.yaml`.
