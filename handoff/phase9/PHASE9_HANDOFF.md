# Phase 9 Handoff: Application Layer Integration

## Deployment Artifact Summary
- **Artifact Name**: `deployment.pt`
- **Export Format**: TorchScript (`torch.jit.script`)

## Validation Summary
- **Numerical Equivalence**: PASSED (Max diff: 0.00000000 <= 1e-05) - Identical output to training checkpoint
- **TorchScript Validation**: PASSED
- **Inference Latency**: 4.65 ms (batch size = 1)

## Integration Instructions

The `CrowdFlowPipeline` now fully supports the deployment artifact via the environment configuration.

To run the Gradio UI with the real model, set the environment variable pointing to the downloaded `deployment.pt` artifact before starting the application.

### Windows Command Prompt
```cmd
set CROWDDNA_MODEL_PATH=C:\path\to\deployment.pt
python app.py
```

### Windows PowerShell
```powershell
$env:CROWDDNA_MODEL_PATH="C:\path\to\deployment.pt"
python app.py
```

> **Note**: Do not modify `CrowdFlowPipeline`, `SequenceBuffer`, or `InferenceRuntime` during this phase. The inference interfaces are strictly owned by the ML Subsystem and are already fully wired to accept this artifact.
