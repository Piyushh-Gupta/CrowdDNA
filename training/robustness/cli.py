"""
CrowdFlow DNA — Robustness CLI
==============================
Module: training/robustness/cli.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from crowdflow_dna.graph.sequence_dataset import SequenceGraphDataset
from training.robustness.context import EvaluationContext
from training.robustness.datasets import SequenceGraphDatasetProvider
from training.robustness.logger import setup_robustness_logger
from training.robustness.pipeline import EvaluationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrowdDNA Robustness Testing Framework")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to deployment.pt artifact")
    parser.add_argument("--config", type=str, default="experiments/baseline.yaml", help="Path to config yaml")
    parser.add_argument("--output", type=str, default="experiments/robustness", help="Output directory")
    parser.add_argument("--protocol", type=str, default="robustness", choices=["clean", "robustness", "stress", "regression"])
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_robustness_logger(output_dir / "robustness.log")
    logger.info("Initializing Robustness Framework...")
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_dir = Path(config.get("simulation", {}).get("output_dir", "data/simulated"))
    dataset = SequenceGraphDataset(
        root=str(dataset_dir)
    )
    dataset_provider = SequenceGraphDatasetProvider(dataset)
    
    context = EvaluationContext(
        deployment_model_path=Path(args.checkpoint),
        dataset_provider=dataset_provider,
        random_seed=args.seed,
        protocol_name=args.protocol,
        device=device,
        output_directory=output_dir,
        config=config,
    )
    
    pipeline = EvaluationPipeline(context)
    pipeline.run()

if __name__ == "__main__":
    main()
