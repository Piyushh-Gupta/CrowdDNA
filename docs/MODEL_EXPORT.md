# CrowdDNA Model Export

## Overview
This document outlines the pipeline for exporting a trained `CrowdDNAModel` into TorchScript and ONNX formats for deployment in production inference environments.

Because the training model natively processes lists of Python `Data` objects, the export pipeline relies on the pure-tensor `CrowdDNADeploymentModel` adapter.

## Deployment Workflow
The canonical export process involves:
1. Loading the target training checkpoint.
2. Initializing `CrowdDNADeploymentModel`.
3. Extracting the weights from the loaded training state dictionary.
4. Exporting to `TorchScript` (via `.script()` or `.trace()`).
5. Exporting to `ONNX` with dynamic axes.
6. Validating both graphs numerically against the native PyTorch implementation.

## Export Formats

### TorchScript (`deployment.pt`)
TorchScript is the preferred deployment target for PyTorch-native C++ backend servers (like LibTorch or TorchServe).
The pipeline natively attempts `torch.jit.script` since it offers better dynamic flow control capabilities over tracing. If standard static analysis fails, it automatically falls back to `torch.jit.trace`.

### ONNX (`crowddna.onnx`)
ONNX enables inter-framework compatibility (e.g., TensorRT, ONNX Runtime, OpenVINO).
The exported ONNX graph contains dynamic axes for batch sizing and graph density:
- **x**: `(total_nodes, in_channels)`
- **edge_index**: `(2, total_edges)`
- **edge_attr**: `(total_edges, edge_dim)`
- **batch**: `(total_nodes)`
- **seq_lengths**: `(batch_size)`

## Validation
To ensure identical mathematical representations, the exporter pushes identical dummy tensors through:
1. PyTorch Deployment Model
2. TorchScript Graph
3. ONNX Runtime (if installed)

It calculates the maximum absolute difference between the output logits. The export succeeds only if `diff <= 1e-5`.

## Troubleshooting
- **ONNX Runtime Warnings**: If ONNX Runtime is not available locally, it will skip ONNX evaluation. Install `onnxruntime` to enable full end-to-end numerical verification.
- **Dynamic Axes Errors**: Because sequences are padded iteratively via variable lengths, certain older ONNX opsets may fail loop tracing. Ensure Opset >= 17 is utilized.
