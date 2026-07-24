"""
CrowdFlow DNA — Experiment Management & Leaderboard Framework
=============================================================

This module provides the entrypoint to the experiment management subsystem,
which discovers completed runs, aggregates metrics, ranks experiments,
and generates standardized leaderboards (Markdown, JSON, CSV) with plots.

Usage:
    python -m training.compare_experiments --experiments experiments/runs --output experiments/leaderboard
"""
import sys
from training.experiment_management import cli_main

def main():
    cli_main()

if __name__ == "__main__":
    sys.exit(main())
