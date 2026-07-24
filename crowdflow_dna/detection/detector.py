"""Pedestrian detection module for CrowdFlow DNA.

Wraps YOLOv8 nano to detect pedestrians in individual video frames
extracted by the Ingestion module, producing bounding boxes for the
Tracking module.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from crowdflow_dna import config
from crowdflow_dna.errors import ModelInferenceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

# BoundingBox represents a single pedestrian detection.
# Components (all absolute pixel coordinates unless noted):
#   x1, y1  – top-left corner of the bounding box (float)
#   x2, y2  – bottom-right corner of the bounding box (float)
#   confidence – YOLO confidence score in range [0.0, 1.0] (float)
#   class_id   – COCO class index; 0 = person (int)
#
# Downstream consumers (Tracking module) must accept this exact tuple shape.
BoundingBox = Tuple[float, float, float, float, float, int]

# COCO class index for "person". Detections with any other class_id are
# discarded by this module and never passed downstream.
_PERSON_CLASS_ID: int = 0


class Yolov8Detector:
    """Detects pedestrians in individual video frames using YOLOv8 nano.

    Wraps the ultralytics YOLO model, filtering all results to only return
    person detections (class_id == 0) that meet the configured confidence
    threshold. Other classes are silently discarded.

    Example usage::

        detector = Yolov8Detector()
        detections: List[BoundingBox] = detector.detect(frame)
        # detections == [(x1, y1, x2, y2, conf, 0), ...]
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """Initialise the detector and load YOLO model weights.

        Args:
            model_path: Path or hub-name of the YOLO model weights file.
                Defaults to ``'yolov8n.pt'``, which ultralytics downloads
                automatically on first use.

        Raises:
            ModelInferenceError: If the model file cannot be found or
                the weights fail to load.
        """
        self._confidence_threshold: float = float(config.CONFIDENCE_THRESHOLD)

        # PyTorch defaults to weights_only=True which breaks older YOLO weights.
        # We temporarily wrap torch.load to force weights_only=False.
        _original_load = torch.load
        
        def _safe_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        try:
            torch.load = _safe_load
            self._model: YOLO = YOLO(model_path)
        except Exception as exc:
            raise ModelInferenceError(
                f"Failed to load YOLO model from '{model_path}': {exc}"
            ) from exc
        finally:
            torch.load = _original_load

        logger.info("Yolov8Detector initialised with model: %s", model_path)

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect pedestrians in a single video frame.

        Args:
            frame: A BGR image as a NumPy ndarray (H, W, 3), as returned by
                ``VideoIngestor.load()``.

        Returns:
            A list of :data:`BoundingBox` tuples
            ``(x1, y1, x2, y2, confidence, class_id)`` containing only
            person detections (``class_id == 0``) that exceed
            ``config.CONFIDENCE_THRESHOLD``.  Returns an empty list when no
            qualifying pedestrians are present; never raises in that case.

        Raises:
            ModelInferenceError: If the underlying YOLO call raises an
                unexpected exception.
        """
        try:
            results = self._model(frame, verbose=False)
        except Exception as exc:
            raise ModelInferenceError(
                f"YOLO inference failed on the provided frame: {exc}"
            ) from exc

        detections: List[BoundingBox] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Move tensors to CPU and convert to NumPy once per result.
            xyxy: np.ndarray = boxes.xyxy.cpu().numpy()   # shape (N, 4)
            confs: np.ndarray = boxes.conf.cpu().numpy()  # shape (N,)
            clses: np.ndarray = boxes.cls.cpu().numpy()   # shape (N,)

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clses):
                if int(cls_id) != _PERSON_CLASS_ID:
                    continue
                if float(conf) < self._confidence_threshold:
                    continue
                detections.append((
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    int(cls_id),
                ))

        logger.debug("detect(): %d person(s) found in frame", len(detections))
        return detections
