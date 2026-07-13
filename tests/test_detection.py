"""Unit tests for Yolov8Detector (crowdflow_dna/detection/detector.py).

All tests use unittest.mock to avoid downloading YOLO weights or running
real GPU inference. Tests are fast and CI-safe.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from crowdflow_dna import config
from crowdflow_dna.errors import ModelInferenceError

# We patch 'crowdflow_dna.detection.detector.YOLO' so that no real weights
# are loaded when the module is imported during test collection.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_FRAME: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_boxes(
    xyxy: List[List[float]],
    confs: List[float],
    clses: List[float],
) -> MagicMock:
    """Build a mock YOLO boxes object from raw lists."""
    boxes = MagicMock()
    boxes.__len__ = MagicMock(return_value=len(xyxy))
    boxes.xyxy.cpu().numpy.return_value = np.array(xyxy, dtype=np.float32)
    boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=np.float32)
    boxes.cls.cpu().numpy.return_value = np.array(clses, dtype=np.float32)
    return boxes


def _make_result(
    xyxy: List[List[float]],
    confs: List[float],
    clses: List[float],
) -> MagicMock:
    """Build a single YOLO result mock."""
    result = MagicMock()
    result.boxes = _make_boxes(xyxy, confs, clses)
    return result


def _make_empty_result() -> MagicMock:
    """Build a YOLO result mock that has no detections."""
    result = MagicMock()
    result.boxes = _make_boxes([], [], [])
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_yolo_cls():
    """Patch ultralytics.YOLO throughout an entire test.

    Yields the mock YOLO *class* (not instance). Each test configures
    mock_yolo_cls.return_value (the instance) to control inference output.
    """
    with patch("crowdflow_dna.detection.detector.YOLO") as mock_cls:
        yield mock_cls


@pytest.fixture()
def detector(mock_yolo_cls):
    """Return a Yolov8Detector backed by a mocked YOLO model."""
    from crowdflow_dna.detection.detector import Yolov8Detector
    return Yolov8Detector()


# ---------------------------------------------------------------------------
# Model initialisation tests
# ---------------------------------------------------------------------------


def test_detector_initialises_successfully(mock_yolo_cls) -> None:
    """Yolov8Detector.__init__ must succeed when YOLO loads without error."""
    from crowdflow_dna.detection.detector import Yolov8Detector
    d = Yolov8Detector()
    mock_yolo_cls.assert_called_once_with("yolov8n.pt")
    assert d is not None


def test_custom_model_path_passed_to_yolo(mock_yolo_cls) -> None:
    """A custom model_path argument must be forwarded to ultralytics.YOLO."""
    from crowdflow_dna.detection.detector import Yolov8Detector
    Yolov8Detector(model_path="yolov8s.pt")
    mock_yolo_cls.assert_called_once_with("yolov8s.pt")


def test_model_load_failure_raises_model_inference_error(mock_yolo_cls) -> None:
    """If YOLO raises during loading, ModelInferenceError must propagate."""
    from crowdflow_dna.detection.detector import Yolov8Detector
    mock_yolo_cls.side_effect = FileNotFoundError("weights not found")
    with pytest.raises(ModelInferenceError, match="Failed to load YOLO model"):
        Yolov8Detector()


def test_model_load_error_chains_original_exception(mock_yolo_cls) -> None:
    """ModelInferenceError must chain the original cause via __cause__."""
    from crowdflow_dna.detection.detector import Yolov8Detector
    original = RuntimeError("corrupt weights")
    mock_yolo_cls.side_effect = original
    with pytest.raises(ModelInferenceError) as exc_info:
        Yolov8Detector()
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Detection output contract tests
# ---------------------------------------------------------------------------


def test_detect_returns_list(detector, mock_yolo_cls) -> None:
    """detect() must always return a list."""
    mock_yolo_cls.return_value.return_value = [_make_empty_result()]
    result = detector.detect(_DUMMY_FRAME)
    assert isinstance(result, list)


def test_detect_output_tuple_has_six_elements(detector, mock_yolo_cls) -> None:
    """Each BoundingBox tuple must contain exactly 6 elements."""
    mock_yolo_cls.return_value.return_value = [
        _make_result([[10.0, 20.0, 100.0, 200.0]], [0.9], [0.0])
    ]
    detections = detector.detect(_DUMMY_FRAME)
    assert len(detections) == 1
    assert len(detections[0]) == 6


def test_detect_output_types_match_contract(detector, mock_yolo_cls) -> None:
    """BoundingBox fields must have the contracted types: 5 floats, 1 int."""
    mock_yolo_cls.return_value.return_value = [
        _make_result([[10.0, 20.0, 100.0, 200.0]], [0.9], [0.0])
    ]
    x1, y1, x2, y2, conf, cls_id = detector.detect(_DUMMY_FRAME)[0]
    assert isinstance(x1, float)
    assert isinstance(y1, float)
    assert isinstance(x2, float)
    assert isinstance(y2, float)
    assert isinstance(conf, float)
    assert isinstance(cls_id, int)


def test_detect_output_values_correct(detector, mock_yolo_cls) -> None:
    """BoundingBox values must exactly match the simulated detection."""
    mock_yolo_cls.return_value.return_value = [
        _make_result([[15.5, 30.0, 120.0, 250.5]], [0.85], [0.0])
    ]
    x1, y1, x2, y2, conf, cls_id = detector.detect(_DUMMY_FRAME)[0]
    assert x1 == pytest.approx(15.5)
    assert y1 == pytest.approx(30.0)
    assert x2 == pytest.approx(120.0)
    assert y2 == pytest.approx(250.5)
    assert conf == pytest.approx(0.85)
    assert cls_id == 0


# ---------------------------------------------------------------------------
# Person-only filtering tests
# ---------------------------------------------------------------------------


def test_non_person_class_filtered_out(detector, mock_yolo_cls) -> None:
    """Detections with class_id != 0 must be silently discarded."""
    # class 2 = car in COCO
    mock_yolo_cls.return_value.return_value = [
        _make_result([[10.0, 10.0, 50.0, 80.0]], [0.95], [2.0])
    ]
    assert detector.detect(_DUMMY_FRAME) == []


def test_mixed_classes_only_persons_returned(detector, mock_yolo_cls) -> None:
    """Only person detections must survive when mixed with other classes."""
    mock_yolo_cls.return_value.return_value = [
        _make_result(
            [[10.0, 10.0, 50.0, 80.0], [200.0, 100.0, 300.0, 400.0]],
            [0.9, 0.8],
            [0.0, 2.0],  # person, car
        )
    ]
    detections = detector.detect(_DUMMY_FRAME)
    assert len(detections) == 1
    assert detections[0][5] == 0  # class_id must be 0


# ---------------------------------------------------------------------------
# Confidence threshold filtering tests
# ---------------------------------------------------------------------------


def test_low_confidence_detection_filtered_out(detector, mock_yolo_cls) -> None:
    """Detections below CONFIDENCE_THRESHOLD must be discarded."""
    below_threshold = config.CONFIDENCE_THRESHOLD - 0.01
    mock_yolo_cls.return_value.return_value = [
        _make_result([[10.0, 20.0, 100.0, 200.0]], [below_threshold], [0.0])
    ]
    assert detector.detect(_DUMMY_FRAME) == []


def test_detection_at_threshold_passes(detector, mock_yolo_cls) -> None:
    """A detection at exactly CONFIDENCE_THRESHOLD must not be discarded."""
    mock_yolo_cls.return_value.return_value = [
        _make_result(
            [[10.0, 20.0, 100.0, 200.0]],
            [config.CONFIDENCE_THRESHOLD],
            [0.0],
        )
    ]
    detections = detector.detect(_DUMMY_FRAME)
    assert len(detections) == 1


def test_detection_above_threshold_passes(detector, mock_yolo_cls) -> None:
    """A detection above CONFIDENCE_THRESHOLD must be retained."""
    above = config.CONFIDENCE_THRESHOLD + 0.1
    mock_yolo_cls.return_value.return_value = [
        _make_result([[10.0, 20.0, 100.0, 200.0]], [above], [0.0])
    ]
    assert len(detector.detect(_DUMMY_FRAME)) == 1


# ---------------------------------------------------------------------------
# Zero-detection tests (user-requested adjustment #4)
# ---------------------------------------------------------------------------


def test_zero_detections_returns_empty_list(detector, mock_yolo_cls) -> None:
    """When YOLO finds no objects at all, detect() must return [] without raising."""
    mock_yolo_cls.return_value.return_value = [_make_empty_result()]
    detections = detector.detect(_DUMMY_FRAME)
    assert detections == []


def test_zero_detections_no_exception_raised(detector, mock_yolo_cls) -> None:
    """Empty YOLO output must never cause an exception to be raised."""
    mock_yolo_cls.return_value.return_value = [_make_empty_result()]
    try:
        detector.detect(_DUMMY_FRAME)
    except Exception as exc:
        pytest.fail(f"detect() raised unexpectedly on zero detections: {exc}")


def test_none_boxes_returns_empty_list(detector, mock_yolo_cls) -> None:
    """When result.boxes is None, detect() must return [] without raising."""
    result = MagicMock()
    result.boxes = None
    mock_yolo_cls.return_value.return_value = [result]
    assert detector.detect(_DUMMY_FRAME) == []


def test_all_filtered_out_returns_empty_list(detector, mock_yolo_cls) -> None:
    """If all detections are non-person, the output must be an empty list."""
    mock_yolo_cls.return_value.return_value = [
        _make_result(
            [[10.0, 10.0, 50.0, 80.0], [60.0, 70.0, 90.0, 120.0]],
            [0.9, 0.8],
            [1.0, 3.0],  # bicycle, motorcycle — no persons
        )
    ]
    assert detector.detect(_DUMMY_FRAME) == []


# ---------------------------------------------------------------------------
# Inference failure tests
# ---------------------------------------------------------------------------


def test_inference_failure_raises_model_inference_error(
    detector, mock_yolo_cls
) -> None:
    """If the underlying model call raises, ModelInferenceError must propagate."""
    mock_yolo_cls.return_value.side_effect = RuntimeError("GPU OOM")
    with pytest.raises(ModelInferenceError, match="YOLO inference failed"):
        detector.detect(_DUMMY_FRAME)


def test_inference_error_chains_original_exception(
    detector, mock_yolo_cls
) -> None:
    """ModelInferenceError from inference must chain the original exception."""
    original = ValueError("bad frame")
    mock_yolo_cls.return_value.side_effect = original
    with pytest.raises(ModelInferenceError) as exc_info:
        detector.detect(_DUMMY_FRAME)
    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Multiple person detections test
# ---------------------------------------------------------------------------


def test_multiple_persons_all_returned(detector, mock_yolo_cls) -> None:
    """All qualifying person detections must be present in the output."""
    mock_yolo_cls.return_value.return_value = [
        _make_result(
            [
                [10.0, 20.0, 100.0, 200.0],
                [150.0, 30.0, 300.0, 400.0],
                [320.0, 50.0, 480.0, 350.0],
            ],
            [0.95, 0.88, 0.76],
            [0.0, 0.0, 0.0],
        )
    ]
    detections = detector.detect(_DUMMY_FRAME)
    assert len(detections) == 3
    assert all(d[5] == 0 for d in detections)
