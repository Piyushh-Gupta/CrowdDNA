"""
Explainability CLI.

Executes the explainability subsystem.
"""
import argparse
import logging
import torch

from crowdflow_dna.explainability.context import ExplanationContext
from crowdflow_dna.explainability.pipeline import ExplanationPipeline
from crowdflow_dna.inference.runtime import InferenceRuntime

import crowdflow_dna.explainability.explainers.confidence  # noqa: F401
import crowdflow_dna.explainability.explainers.ablation  # noqa: F401
import crowdflow_dna.explainability.explainers.attention  # noqa: F401
import crowdflow_dna.explainability.reporting.markdown  # noqa: F401
import crowdflow_dna.explainability.reporting.json_report  # noqa: F401
import crowdflow_dna.explainability.reporting.csv_report  # noqa: F401
import crowdflow_dna.explainability.plotting.plots  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Explainability subsystem.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to deployment.pt")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset YAML config")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--explainer", type=str, nargs="+", default=["confidence", "ablation", "attention"], help="Explainers to run")
    parser.add_argument("--all", action="store_true", help="Run all available explainers")
    parser.add_argument("--force", action="store_true", help="Force ignore cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading deployment model from {args.checkpoint}")
    runtime = InferenceRuntime()
    try:
        runtime.load_model(args.checkpoint)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import sys
        sys.exit(1)
        
    if not hasattr(runtime.backend, "model") or runtime.backend.model is None:
        logger.error("Explainability framework requires a PyTorch/TorchScript model (.pt).")
        import sys
        sys.exit(1)
        
    model = runtime.backend.model
    
    from torch_geometric.data import Data
    
    class MockDataset:
        def __iter__(self):
            # Create a valid input for CrowdDNADeploymentModel
            data = Data(
                x=torch.randn(10, 5),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 4), dtype=torch.float),
                batch=torch.arange(10, dtype=torch.long),
                seq_lengths=torch.tensor([10])
            )
            # Yield (sequence_id, data)
            yield "seq_1", data
            yield "seq_2", data
            
    dataset = MockDataset()
    
    explainers = args.explainer
    from crowdflow_dna.explainability.registry import ExplainerRegistry
    if args.all:
        explainers = list(ExplainerRegistry.list_explainers().keys())
    else:
        valid_explainers = set(ExplainerRegistry.list_explainers().keys())
        for exp in explainers:
            if exp not in valid_explainers:
                logger.error(f"Unsupported explainer: '{exp}'. Available: {', '.join(valid_explainers)}")
                import sys
                sys.exit(1)
        
    context = ExplanationContext(
        model=model,
        runtime=runtime,
        dataset_provider=dataset,
        output_directory=args.output,
        configuration={"explainers": explainers, "force": args.force},
        logger=logger,
        device=device
    )
    
    pipeline = ExplanationPipeline(output_dir=args.output)
    pipeline.run(context)
    logger.info(f"Outputs written to {args.output}")

if __name__ == "__main__":
    main()
