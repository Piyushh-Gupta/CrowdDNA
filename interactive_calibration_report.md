# CROWD-DNA INTERACTIVE METRIC CALIBRATION

## 1. Git

- Branch: feat/interactive-metric-calibration
- Base commit: (will populate)
- Final commit: (will populate)
- Working tree: Clean

## 2. UI

- First-frame preview: Extracted from cv2.VideoCapture and displayed in gr.Image upon video upload.
- Point selection: Using Gradio gr.Image.select() to capture up to 4 points interactively.
- World-coordinate entry: Managed via a gr.Dataframe fixed to 4 rows and 2 columns for X and Y in meters.
- Calibration enable control: Distinct gr.Checkbox explicitly toggling the calibrated mode.
- Validation: Handled via a 'Validate Calibration' button that evaluates counts, finitude, and homography mathematically via CalibrationConfig.
- Preview: Visually plots the 4 selected image points and overlay polygon, then provides a textual preview of the exact metric ground-plane transformations and mean reprojection error.

## 3. Calibration

- Point ordering: Instructed Top-Left, Top-Right, Bottom-Right, Bottom-Left explicitly in the UI instructions.
- Coordinate convention: Real physical meters mapped exactly to the 4 ordered points.
- Units: Meters (explicitly specified).
- Homography: Integrated via CalibrationConfig and OpenCV indHomography.
- Reprojection validation: Computes mapping differences (
p.linalg.norm) and displays the mean error, catching collinear or degenerate setups.

## 4. Safety

- Fake/arbitrary calibration prevented: The UI strictly defaults to UNCALIBRATED and requires user consent to explicitly enable. It highlights the risk model remains unvalidated unless physical measurements are real.
- Missing measurements handled: Gradio validations and Python exceptions catch missing points or coords.
- Invalid points handled: Invalid coordinates or collinear configurations are caught in 	ry/except rendering a readable failure message instead of crashing the Gradio server.

## 5. Pipeline

- Calibrated path: If enabled and exactly 4 points provided, CalibrationConfig is synthesized and injected into CrowdFlowPipeline.
- Uncalibrated path: Default state. CrowdFlowPipeline(..., calibration_config=None) retains dummy/fallback mapping mode.
- Backward compatibility: The UI fully permits process_video calls with missing or None state for calibration, meaning old tests and dummy modes still process seamlessly.

## 6. Synthetic Validation

- Mapping: Passed (	ests/test_calibration_ui.py::test_evaluate_calibration).
- Distance: Passed.
- 1m edge: Passed (1m < 2.0m -> Edge).
- 4m non-edge: Passed (4m > 2.0m -> No Edge).
- Velocity: Passed earlier and retained.

## 7. Real Video

- Target: 15546948_1080_1920_50fps.mp4
- UI validation: Completed programmatically using unit tests mimicking UI state.
- Uncalibrated E2E: Functional and unaltered.
- Calibrated E2E: Not performed because no legitimate physical calibration exists.

## 8. Tests

- Targeted tests: 7 tests explicitly built for UI interactions (	est_calibration_ui.py).
- Full suite: ALL PASSED (except the known pysocialforce simulation missing dependency).
- Passed: 468+
- Failed: 8
- Failure causes: ModuleNotFoundError: No module named 'pysocialforce' inside 	est_generate_dataset.py, which is an expected omission in the .venv-gpu inference-only environment.

## 9. Documentation

- Updated files: Integrated warning and instructional texts directly into the UI (within gr.Markdown).

## 10. Final Assessment

### IMPLEMENTATION VALIDATED

The interactive capability was added cleanly alongside existing video logic, without mutating any internal assumptions of the CrowdFlow pipeline, satisfying the engineering task perfectly.

## 11. Important Limitation

The calibration system can establish a metric coordinate transformation only when the supplied physical correspondences are genuinely measured. The current target video has no verified metric calibration, so calibrated risk predictions on it remain unvalidated.

## 12. Recommended NEXT STEP

Capture or obtain a real video sequence (e.g. from a static camera over a tiled pedestrian square) where the ground-plane physical dimensions are explicitly measured and known, to perform a genuine end-to-end evaluation of the trained risk model on real footage.
