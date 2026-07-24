# CrowdDNA Baseline Results (v1.0.0)

This document establishes the canonical benchmark for the CrowdDNA ML pipeline. 
All future phases must use these metrics as the baseline for performance and accuracy comparisons.

---

## 1. Dataset Statistics

- **Dataset Path**: `{dataset_path}`
- **Total Sequences**: {total_sequences}
- **Total Frames/Graphs**: {total_frames}
- **Average Sequence Length**: {average_sequence_length}
- **Sequence Length Range**: [{min_sequence_length}, {max_sequence_length}]
- **Average Nodes per Frame**: {average_nodes_per_frame}
- **Average Edges per Frame**: {average_edges_per_frame}

### Class Distribution
- **Safe**: {class_safe}
- **Congested**: {class_congested}
- **Critical**: {class_critical}

### Issues
- **Empty Graphs**: {empty_graph_count}
- **Invalid Graphs**: {invalid_graph_count}
- **Missing Labels**: {missing_labels}

---

## 2. Training

- **Epochs**: {epochs}
- **Best Epoch**: {best_epoch}
- **Total Training Time**: {total_training_time}

---

## 3. Evaluation Metrics

- **Accuracy**: {accuracy}
- **Macro Precision**: {precision_macro}
- **Macro Recall**: {recall_macro}
- **Macro F1**: {f1_macro}

### Class Performance (F1-Score)
- **Safe**: {f1_safe}
- **Congested**: {f1_congested}
- **Critical**: {f1_critical}

---

## 4. Deployment Validation

- **Validation Success**: {validation_success}
- **Max Output Difference (TorchScript vs PyTorch)**: {max_output_difference}

### Model Complexity & Sizes
- **Parameter Count**: {parameter_count}
- **PyTorch Checkpoint**: {pt_size} MB
- **TorchScript Size**: {ts_size} MB
- **ONNX Size**: {onnx_size} MB

### Inference Latency (Batch Size = 1)
- **TorchScript via InferenceRuntime**: {ts_latency} ms
- **ONNX via InferenceRuntime**: {onnx_latency} ms

---

## 5. Environment & Reproducibility

- **Git Tag**: {git_tag}
- **Commit Hash**: {git_commit}
- **Configuration Snapshot**:
```yaml
{configuration_snapshot}
```
