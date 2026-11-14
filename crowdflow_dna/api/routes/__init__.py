"""CrowdDNA API routes package."""

from crowdflow_dna.api.routes.inference import router as inference_router
from crowdflow_dna.api.routes.system import router as system_router

__all__ = ["inference_router", "system_router"]
