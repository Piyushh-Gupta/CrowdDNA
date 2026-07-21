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
