# Explainability & Model Interpretability Framework

## Overview
The Explainability Framework allows developers and researchers to analyze how CrowdDNA models make their predictions. It provides deep insights into the most important nodes (pedestrians), interactions (edges), and frames (temporal importance).

## Architecture
The subsystem is housed in `crowdflow_dna/explainability/` and orchestrates:
- **ExplanationContext**: Immutable state definition.
- **ExplanationSession**: Traceability and global session metadata.
- **ExplanationGraph**: Canonical exchange object.
- **ExplanationEngine**: Algorithm execution.
- **ExplanationPipeline**: Orchestration and report generation.
- **Registry**: Metadata-driven loading of algorithms, plotters, and reporters.

## Explanation Algorithms
1. **Confidence Explainer**: Analyzes raw logits, entropy, and prediction margins.
2. **Ablation Explainer**: Systematically ablates nodes to measure probability drops, identifying critical actors in the scene.
3. **Attention Explainer**: Gracefully attempts to extract GAT and Temporal attention weights.

## Output Formats
- **Reports**: Markdown, JSON, and CSV reports are written to the output directory.
- **Plots**: `node_importance` and `confidence` plots are saved as PNGs.

## Caching
The framework caches expensive explanation computations (like ablation) under `experiments/explainability/cache/`. Caching is based on a deterministic hash of the model, dataset, explainer, and configuration.

## CLI Usage
Run the explainability framework using:

```bash
python -m training.run_explainability \
    --checkpoint path/to/deployment.pt \
    --dataset configs/my_dataset.yaml \
    --output experiments/explainability/run_1 \
    --explainer confidence ablation attention
```

## Extension Guide
To add a new explainer:
1. Create a class inheriting from `ExplainerProtocol`.
2. Decorate it with `@ExplainerRegistry.register(...)` and provide `ExplainerMetadata`.
3. Implement the `explain()` method to return an `ExplanationGraph`.
