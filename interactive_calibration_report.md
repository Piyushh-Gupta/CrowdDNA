# CROWD-DNA INTERACTIVE METRIC CALIBRATION

## 1. Implementation status
Implemented / structurally validated.

## 2. What is validated
- UI workflow;
- point selection;
- world-coordinate input;
- homography construction;
- geometric validation;
- synthetic held-out mapping test;
- pipeline configuration propagation;
- uncalibrated backward compatibility.

## 3. What is NOT validated
- real-world accuracy of any calibration without measured physical landmarks;
- real-world risk-model accuracy on the current target video;
- domain-shift performance;
- model generalization.

## 4. Important clarification
A four-point homography fit residual is NOT an independent accuracy measurement.
The displayed 'fit residual' merely measures how well the homography model satisfies the 4 input points mathematically.
It does NOT prove the calibration is accurate for real-world distances. Real-world accuracy requires independent held-out physical measurements which are not available for the stock target video.

## 5. UI and Architecture Enhancements
- Introduced explicit point validation (geometry_validation.py) to reject collinear, self-intersecting, and numerically unstable quadrilateral configurations.
- Synthetic validation added (	est_calibration_synthetic.py) to confirm mapping math works on independent/held-out synthetic coordinates.
- Gradio state management correctly invalidates prior successful calibrations when coordinates are modified, ensuring no stale configurations reach the CrowdFlowPipeline.
- Gradio event wiring integration verified successfully through 	est_calibration_integration.py passing synthetic metric inputs all the way to MetricCalibrator.
## 6. Test suite
- targeted tests (tests/test_calibration*.py): ALL PASSED
- full suite in .venv-gpu (pytest -v): 
  - 479 passed
  - 8 failed
  - 9 skipped
- Failure names:
  - test_creates_one_json_per_run
  - test_creates_manifest_json
  - test_manifest_has_correct_entry_count
  - test_manifest_covers_all_risk_classes
  - test_each_json_has_correct_schema
  - test_deterministic_output
  - test_seeds_are_unique_across_all_runs
  - test_num_timesteps_in_output_matches_config
- Failure root causes: ModuleNotFoundError: No module named 'pysocialforce' inside 	est_generate_dataset.py. This is an expected environment dependency issue because the .venv-gpu environment is dedicated strictly to inference and does not install simulation packages.
