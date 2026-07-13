"""Detection package for CrowdFlow DNA.

This package provides pedestrian detection using YOLOv8 nano.
It is the second stage in the pipeline, consuming frames from the
Ingestion module and producing bounding boxes for the Tracking module.
"""

from crowdflow_dna.detection.detector import BoundingBox, Yolov8Detector

__all__ = ["Yolov8Detector", "BoundingBox"]
