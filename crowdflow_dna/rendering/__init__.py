"""Rendering package for CrowdFlow DNA.

Provides frame annotation and risk timeline accumulation.
This is the sixth stage in the pipeline, consuming TrackItem and
RiskPrediction objects to produce annotated frames and a structured
timeline for the Gradio UI.
"""

from crowdflow_dna.rendering.timeline import TimelineBuilder, TimelineEntry
from crowdflow_dna.rendering.video_renderer import FrameAnnotator

__all__ = ["FrameAnnotator", "TimelineBuilder", "TimelineEntry"]
