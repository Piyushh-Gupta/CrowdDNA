# Inference Runtime

The Inference Runtime is the canonical deployment entry point for Aayushi's pipeline. It provides a unified abstraction for executing models exported from the `CrowdDNA` framework.

## Supported Backends

The runtime automatically determines the appropriate backend based on the model file's extension:
- **TorchScript** (`.pt`): Evaluated natively on GPU if available, falling back to CPU.
- **ONNX Runtime** (`.onnx`): Evaluated via `onnxruntime` utilizing the `CPUExecutionProvider` for cross-platform robustness.

## Loading Workflow

Instantiate an `InferenceRuntime` and call `load_model(path)`:
```python
from crowdflow_dna.inference import InferenceRuntime

runtime = InferenceRuntime()
runtime.load_model("exports/deployment.pt")
```

The runtime will instantiate the correct backend, parse the model, and prepare it for inference.
If the path does not exist or the format is unsupported, it will raise explicit exceptions (`ModelNotFoundError`, `UnsupportedModelFormatError`).

## Runtime API

### `predict(x, edge_index, edge_attr, batch, seq_lengths) -> InferenceResult`

Performs inference on a graph and returns a standard `InferenceResult` dataclass.

### `predict_batch(x, edge_index, edge_attr, batch, seq_lengths) -> list[InferenceResult]`

Performs inference on a batch of temporal sequences and returns a list of results.

### `InferenceResult`

A dataclass representing the model prediction containing:
- `predicted_class`: Integer index of the highest probability class.
- `probabilities`: A NumPy `ndarray` containing the full probability distribution.
- `confidence`: The raw float probability of the predicted class.
- `backend`: A string describing the active backend (e.g., `"TorchScript"`).
- `inference_time_ms`: Time in milliseconds to compute inference.
- `model_format`: Format detected during load (e.g., `"TorchScript"`, `"ONNX"`).
- `model_version`: Optional version identifier.

## Expected Tensor Signatures

Since the runtime bypasses PyTorch Geometric `Data` objects entirely to maintain compatibility with TorchScript and ONNX, it requires raw flattened PyTorch `Tensor` inputs representing the batch:
- `x`: Node features `(total_nodes, num_node_features)`
- `edge_index`: Edges `(2, total_edges)`
- `edge_attr`: Edge features `(total_edges, num_edge_features)`
- `batch`: Node-to-graph assignment mapping `(total_nodes,)`
- `seq_lengths`: Length of each temporal trajectory sequence `(batch_size,)`

## SequenceBuffer Edge Cases (Empty Frames)

The inference pipeline's `SequenceBuffer` is responsible for concatenating multiple temporal graph frames into a unified `TensorBatch`.

When processing video streams, it is common to encounter empty frames (e.g., frames with zero detected people). If an empty frame contains zero nodes, PyTorch Geometric's downstream spatial pooling functions (like `global_mean_pool`) can lose track of the original sequence length because `global_mean_pool` defaults its dimensionality to `batch.max() + 1`. This leads to a dimension mismatch between the expected `sum(seq_lengths)` and the pooled node embeddings, causing a fatal out-of-bounds segfault inside the exported TorchScript C++ runtime.

To guarantee that the runtime receives structurally valid tensors:
1. `SequenceBuffer` intercepts any frame containing `0` nodes.
2. It transparently inserts a **dummy node** (`[0.0, 0.0, 0.0, ...]` across all node feature dimensions) for that frame.
3. This dummy node is registered in the `batch` tensor.

**Why does this not change model outputs?**
By inserting an all-zeros dummy node with no edges, the `GATConv` layers pass the zeros through cleanly. The `global_mean_pool` subsequently averages a single all-zeros vector to produce an all-zeros graph embedding for that specific frame. Functionally, this is mathematically identical to what PyTorch Geometric naturally produces for an empty graph during standard training.

**Why does the deployment model remain valid without retraining?**
Because the spatial and temporal arithmetic remains identical, the TorchScript artifact (`deployment.pt`) does not need to be rebuilt or retrained. It still processes the identical tensor structures, simply bypassing a low-level C++ indexing limitation by keeping the `batch` tensor contiguous and aligned with `seq_lengths`.
