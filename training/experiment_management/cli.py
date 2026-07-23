"""
CLI orchestrator for the experiment management subsystem.
"""
import argparse
import logging
from pathlib import Path

from training.experiment_management.discovery import ExperimentDiscovery
from training.experiment_management.loader import ExperimentLoader
from training.experiment_management.ranking import STRATEGIES, DefaultRankingStrategy
from training.experiment_management.comparator import ExperimentComparator
from training.experiment_management.leaderboard import LeaderboardGenerator
from training.experiment_management.plotting import PlotGenerator

logger = logging.getLogger(__name__)


def cli_main() -> None:
    """Main CLI entrypoint for the experiment comparator."""
    parser = argparse.ArgumentParser(description="CrowdDNA Experiment Comparator and Leaderboard Generator")
    parser.add_argument("--experiments", type=str, default="experiments/runs", help="Base directory containing experiment runs")
    parser.add_argument("--output", type=str, default="experiments/leaderboard", help="Output directory for the leaderboard")
    parser.add_argument("--top", type=int, default=None, help="Limit leaderboard to top N experiments")
    parser.add_argument("--sort-by", type=str, default="default", choices=list(STRATEGIES.keys()), help="Ranking strategy to use")
    parser.add_argument("--include-incomplete", action="store_true", help="Include incomplete/failed runs in discovery")
    parser.add_argument("--format", type=str, default="md,csv,json", help="Comma-separated list of formats to generate")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    logger.info("Initializing Experiment Management Subsystem...")
    
    # 1. Setup
    exp_dir = Path(args.experiments)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    discovery = ExperimentDiscovery(base_dir=str(exp_dir))
    loader = ExperimentLoader(discovery)
    
    strategy = STRATEGIES.get(args.sort_by, DefaultRankingStrategy())
    comparator = ExperimentComparator(strategy)
    
    # 2. Load Experiments
    logger.info(f"Discovering experiments in {exp_dir}...")
    experiments = loader.load_all(include_incomplete=args.include_incomplete)
    
    if not experiments:
        logger.warning("No valid experiments found.")
        return
        
    logger.info(f"Successfully loaded {len(experiments)} valid experiments.")
    
    # 3. Compare and Rank
    logger.info("Ranking experiments...")
    ranked_exps = comparator.compare(experiments, top_n=args.top)
    
    # 4. Generate Outputs
    formats = [fmt.strip().lower() for fmt in args.format.split(",")]
    
    logger.info(f"Generating leaderboards in {out_dir} (Formats: {formats})...")
    
    if "json" in formats:
        LeaderboardGenerator.generate_json(ranked_exps, out_dir)
    if "csv" in formats:
        LeaderboardGenerator.generate_csv(ranked_exps, out_dir)
    if "md" in formats:
        LeaderboardGenerator.generate_markdown(ranked_exps, out_dir)
        
    # 5. Generate Plots
    logger.info("Generating comparison plots...")
    PlotGenerator.generate_plots(ranked_exps, out_dir)
    
    logger.info(f"Done. Best model is {ranked_exps[0].name}.")
