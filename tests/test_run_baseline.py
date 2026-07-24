import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from training.run_baseline import main


@pytest.fixture
def mock_runner(tmp_path):
    with patch("training.run_baseline.ExperimentRunner") as MockRunner:
        instance = MockRunner.return_value
        
        # Setup mock result
        mock_result = MagicMock()
        mock_result.best_checkpoint_path = "/fake/best.pt"
        mock_result.total_training_time = 1.0
        mock_result.evaluation_time = 1.0
        mock_result.evaluation_result.accuracy = 1.0
        mock_result.evaluation_result.precision_macro = 1.0
        mock_result.evaluation_result.recall_macro = 1.0
        mock_result.evaluation_result.f1_macro = 1.0
        mock_result.experiment_directory = tmp_path / "baseline_test"
        
        # Fake metadata file creation from ArtifactManager
        meta_dir = tmp_path / "baseline_test" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "experiment_metadata.json", "w") as f:
            json.dump({"timestamp": "test"}, f)
            
        instance.run.return_value = mock_result
        yield MockRunner


def test_run_baseline_success(mock_runner, capsys, tmp_path):
    exit_code = main()
    assert exit_code == 0
    
    captured = capsys.readouterr()
    assert "CrowdDNA Baseline Workflow Complete" in captured.out
    assert "Accuracy:" in captured.out
    assert "F1:" in captured.out


def test_run_baseline_failure(mock_runner, caplog):
    mock_runner.return_value.run.side_effect = RuntimeError("Training failed")
    
    with caplog.at_level(logging.ERROR):
        exit_code = main()
        
    assert exit_code == 1
    assert "Baseline workflow failed." in caplog.text


def test_run_baseline_missing_config():
    with patch("training.run_baseline.Path.exists", return_value=False):
        exit_code = main()
        assert exit_code == 1
