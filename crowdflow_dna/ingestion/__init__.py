"""Ingestion package for CrowdFlow DNA.

This package provides the entry point for ingesting and validating
video files before they are processed by the detection module.
"""

from crowdflow_dna.ingestion.video_loader import VideoIngestor

__all__ = ["VideoIngestor"]
