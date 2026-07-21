
import numpy as np
import pytest

from training.evaluate_model import EvaluationResult
from training.reporting import ReportGenerator, ReportArtifacts


@pytest.fixture
def mock_evaluation_result():
    # 3 classes: "Safe", "Congesting", "Critical"
    targets = np.array([0, 1, 1, 2, 0])
    predictions = np.array([0, 1, 0, 2, 2])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.1, 0.8],
        [0.3, 0.2, 0.5]
    ])
    
    cm = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])
    
    cm_norm = np.array([
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return EvaluationResult(
        accuracy=0.6,
        precision_macro=0.6,
        recall_macro=0.6,
        f1_macro=0.6,
        precision_per_class=np.array([0.5, 1.0, 0.5]),
        recall_per_class=np.array([0.5, 0.5, 1.0]),
        f1_per_class=np.array([0.5, 0.66, 0.66]),
        confusion_matrix=cm,
        normalized_confusion_matrix=cm_norm,
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
    )


@pytest.fixture
def missing_class_result():
    # Only 0 and 1 exist in targets. 2 is missing.
    targets = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 2])
    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7]
    ])
    
    cm = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 0, 0]
    ])
    
    cm_norm = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0]
    ])
    
    return EvaluationResult(
        accuracy=0.75,
        precision_macro=0.75,
        recall_macro=0.75,
        f1_macro=0.75,
        precision_per_class=np.array([1.0, 1.0, 0.0]),
        recall_per_class=np.array([1.0, 0.5, 0.0]),
        f1_per_class=np.array([1.0, 0.66, 0.0]),
        confusion_matrix=cm,
        normalized_confusion_matrix=cm_norm,
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
    )


def test_reporting_standard_generation(tmp_path, mock_evaluation_result):
    classes = ["Safe", "Congesting", "Critical"]
    gen = ReportGenerator(tmp_path, classes)
    
    meta = {
        "timestamp": "2023-01-01T00:00:00Z",
        "git_commit": "abcdef",
        "total_training_time": 10.5,
        "evaluation_time": 1.2,
        "best_checkpoint_path": "/fake/best.pt",
        "configuration_snapshot": {"test": True}
    }
    
    artifacts = gen.generate(mock_evaluation_result, meta)
    
    assert isinstance(artifacts, ReportArtifacts)
    
    # Check paths generated
    assert artifacts.confusion_matrix_png.exists()
    assert artifacts.normalized_confusion_matrix_png.exists()
    assert artifacts.roc_curve_png.exists()
    assert artifacts.precision_recall_curve_png.exists()
    assert artifacts.metrics_csv.exists()
    assert artifacts.metrics_json.exists()
    assert artifacts.summary_markdown.exists()
    
    # Check CSV contents
    with open(artifacts.metrics_csv, "r") as f:
        lines = f.readlines()
        assert len(lines) == 4 # Header + 3 classes
        assert "Safe" in lines[1]
        assert "Congesting" in lines[2]
        
    # Check Markdown
    with open(artifacts.summary_markdown, "r") as f:
        content = f.read()
        assert "abcdef" in content # git commit
        assert "0.6000" in content # Accuracy
        assert "roc_curve.png" in content


def test_missing_class_graceful_handling(tmp_path, missing_class_result, caplog):
    classes = ["Safe", "Congesting", "Critical"]
    gen = ReportGenerator(tmp_path, classes)
    
    # Should not throw any exception when processing roc/pr curves
    artifacts = gen.generate(missing_class_result, {})
    
    assert artifacts.roc_curve_png.exists()
    assert artifacts.precision_recall_curve_png.exists()
    
    assert "has no positive samples. Skipping ROC curve" in caplog.text
    assert "has no positive samples. Skipping PR curve" in caplog.text


def test_empty_probability_matrix(tmp_path, mock_evaluation_result):
    classes = ["Safe", "Congesting", "Critical"]
    gen = ReportGenerator(tmp_path, classes)
    
    # Emptify probabilities
    empty_res = EvaluationResult(
        accuracy=0.0,
        precision_macro=0.0,
        recall_macro=0.0,
        f1_macro=0.0,
        precision_per_class=np.array([]),
        recall_per_class=np.array([]),
        f1_per_class=np.array([]),
        confusion_matrix=np.array([]),
        normalized_confusion_matrix=np.array([]),
        predictions=np.array([]),
        targets=np.array([]),
        probabilities=np.array([]),
    )
    
    with pytest.raises(ValueError, match="Empty probability matrix"):
        gen.generate(empty_res, {})
