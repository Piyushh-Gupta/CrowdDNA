"""Tracking package for CrowdFlow DNA.

This package provides multi-object tracking using an IoU-based
association strategy. It is the third stage in the pipeline,
consuming BoundingBox detections from the Detection module and
producing TrackItem objects for the Pipeline orchestrator.
"""

from crowdflow_dna.tracking.tracker import ByteTracker

__all__ = ["ByteTracker"]
