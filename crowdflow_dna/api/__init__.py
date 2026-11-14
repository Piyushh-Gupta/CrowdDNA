"""CrowdDNA REST API package.

Provides the FastAPI application factory, system endpoints,
and common schemas for real-time crowd dynamics analysis.
"""

from crowdflow_dna.api.app import create_app

__all__ = ["create_app"]
