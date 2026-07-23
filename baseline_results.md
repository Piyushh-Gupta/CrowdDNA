# CrowdDNA Baseline Results (v1.0.0)

This document establishes the canonical benchmark for the CrowdDNA ML pipeline. 
All future phases must use these metrics as the baseline for performance and accuracy comparisons.

---

## 1. Dataset Statistics

- **Dataset Path**: `data\simulated`
- **Total Sequences**: 300
- **Total Frames/Graphs**: 90000
- **Average Sequence Length**: 300.0
- **Sequence Length Range**: [300, 300]
- **Average Nodes per Frame**: 35.0
- **Average Edges per Frame**: 193.46

### Class Distribution
- **Safe**: 54569
- **Congested**: 0
- **Critical**: 1523

### Issues
- **Empty Graphs**: 0
- **Invalid Graphs**: 0
- **Missing Labels**: 0

---

## 2. Training

- **Epochs**: 0
- **Best Epoch**: 10
- **Total Training Time**: See run_baseline logs

---

## 3. Evaluation Metrics

- **Accuracy**: 1.0
- **Macro Precision**: 1.0
- **Macro Recall**: 1.0
- **Macro F1**: 1.0

### Class Performance (F1-Score)
- **Safe**: 1.0
- **Congested**: 1.0
- **Critical**: 1.0

---

## 4. Deployment Validation

- **Validation Success**: True
- **Max Output Difference (TorchScript vs PyTorch)**: 0.0

### Model Complexity & Sizes
- **Parameter Count**: 422278
- **PyTorch Checkpoint**: 4.86 MB
- **TorchScript Size**: 1.71 MB
- **ONNX Size**: 0.0 MB

### Inference Latency (Batch Size = 1)
- **TorchScript via InferenceRuntime**: 3.75 ms
- **ONNX via InferenceRuntime**: 0.0 ms

---

## 5. Environment & Reproducibility

- **Git Tag**: v1.0.0
- **Commit Hash**: 6fdc5b0558c9bd9fe4483be0cb7d63b4d764d531
- **Configuration Snapshot**:
```yaml
model:
  classes:
  - Safe
  - Congesting
  - Critical
  gnn_hidden_dim: 128
  gru_bidirectional: true
  gru_dropout: 0.1
  gru_hidden_dim: 64
  gru_num_layers: 2
  num_gnn_layers: 2
simulation:
  output_dir: data/simulated
training:
  batch_size: 16
  checkpoint_dir: C:\Users\piyus\Downloads\CrowdDNA\experiments\runs\baseline_20260722_230537\checkpoints
  deterministic: true
  learning_rate: 0.001
  num_epochs: 50
  random_seed: 42
  validation_split: 0.2

```
