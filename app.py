"""Gradio UI entry point for CrowdFlow DNA Phase 8.

Provides an interface to upload a video, run the CrowdFlowPipeline,
and display the annotated video, risk timeline, and metadata.

Set the ``CROWDDNA_MODEL_PATH`` environment variable to the path of an
exported ``.pt`` or ``.onnx`` deployment model to enable inference mode.
When the variable is absent, the pipeline runs in dummy mode.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import gradio.networking
import gradio_client.utils as client_utils

from crowdflow_dna.errors import CrowdFlowError
from crowdflow_dna.rendering.timeline import TimelineEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkeypatch gradio_client to prevent schema generation crash with Pydantic 2.x booleans
orig_schema_to_python = client_utils._json_schema_to_python_type

def _patched_schema_to_python(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    if isinstance(schema, dict):
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], bool):
            schema["additionalProperties"] = {} if schema["additionalProperties"] else {"type": "null"}
        if "items" in schema and isinstance(schema["items"], bool):
            schema["items"] = {} if schema["items"] else {"type": "null"}
        if "prefixItems" in schema and isinstance(schema["prefixItems"], bool):
            schema["prefixItems"] = {} if schema["prefixItems"] else {"type": "null"}
    return orig_schema_to_python(schema, defs)

client_utils._json_schema_to_python_type = _patched_schema_to_python





def _timeline_to_dataframe(timeline: List[TimelineEntry]) -> list:
    """Convert timeline entries to a list of lists for gr.DataFrame.

    If a frame has no predictions (dummy mode), outputs a row with
    sentinel values indicating no model was used.
    """
    rows = []
    for entry in timeline:
        if not entry.predictions:
            rows.append([entry.frame_index, "—", "No model", "—"])
        else:
            for pred in entry.predictions:
                rows.append(
                    [
                        entry.frame_index,
                        pred.region_id,
                        pred.label,
                        f"{pred.confidence:.2f}",
                    ]
                )
    return rows


def _metadata_to_rows(metadata: Dict[str, Any]) -> list:
    """Convert metadata dict to key-value rows for gr.DataFrame."""
    if not metadata:
        return []
    
    return [
        ["FPS", f"{metadata.get('fps', 0):.2f}"],
        ["Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"],
        ["Total Frames", metadata.get("frame_count", 0)],
        ["Duration (s)", f"{metadata.get('duration_seconds', 0):.2f}"],
        ["Sample Rate", metadata.get("sample_rate", 1)],
    ]



import cv2
import numpy as np
from crowdflow_dna.calibration import CalibrationConfig
from crowdflow_dna.calibration_utils import draw_calibration_points, evaluate_calibration

def get_first_frame(video_file: Optional[str]) -> Tuple[Optional[np.ndarray], list, str, Optional[np.ndarray]]:
    if not video_file:
        return None, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "No video", None
    
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "Error reading video", None
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    df = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    msg = "Frame loaded. Click 4 points in order (Top-Left, Top-Right, Bottom-Right, Bottom-Left)"
    
    return frame_rgb, df, msg, frame_rgb

def handle_image_click(evt: gr.SelectData, image: np.ndarray, state_pts: list):
    x, y = evt.index
    if len(state_pts) >= 4:
        state_pts = [] # Reset if already 4 points
        
    state_pts.append((x, y))
    
    drawn_img = draw_calibration_points(image, state_pts)
    
    msg = f"Selected {len(state_pts)}/4 points."
    if len(state_pts) == 4:
        msg += " Now enter the real-world coordinates in meters."
        
    return drawn_img, state_pts, msg

def validate_and_preview(state_pts: list, world_coords: list, enable_calib: bool):
    if not enable_calib:
        return "UNCALIBRATED", "Calibration disabled.", gr.update()
        
    if len(state_pts) != 4:
        return "UNCALIBRATED", f"Need exactly 4 image points, got {len(state_pts)}", gr.update()
        
    try:
        w_pts = []
        for row in world_coords:
            w_pts.append((float(row[0]), float(row[1])))
            
        cfg = CalibrationConfig(enabled=True, image_points=state_pts, world_points_m=w_pts)
        res = evaluate_calibration(cfg)
        
        err = res["mean_error"]
        status_msg = f"CALIBRATION VALID. Mean reprojection error: {err:.4f} m"
        state_val = "CALIBRATION ENABLED"
        
        preview_text = "Metric Ground-Plane Preview:\n"
        for i, pt in enumerate(res["transformed_points"]):
            preview_text += f"Point {i+1}: {pt[0]:.2f}m, {pt[1]:.2f}m\n"
            
        return state_val, status_msg, gr.update(value=preview_text)
    except Exception as e:
        return "UNCALIBRATED", f"Invalid Calibration: {e}", gr.update(value="")

def process_video(
    video_file: Optional[str],
    state_pts: list = None,
    world_coords: list = None,
    enable_calib: bool = False
) -> Tuple[Optional[str], list, list, str]:
    """Gradio callback to run the pipeline on an uploaded video.

    Args:
        video_file: Path to the uploaded video file provided by Gradio.

    Returns:
        Tuple of (output_video_path, timeline_rows, metadata_rows, status_msg).
    """
    thread_id = __import__('threading').get_ident()
    logger.info("[Thread %s] Entering app.process_video()", thread_id)
    
    from crowdflow_dna.inference import ModelNotFoundError, UnsupportedModelFormatError
    from crowdflow_dna.pipeline import CrowdFlowPipeline
    
    if not video_file:
        logger.info("[Thread %s] process_video() return - no video", thread_id)
        return None, [], [], "Please upload a video file."

    logger.info("[Thread %s] Processing uploaded video: %s", thread_id, video_file)


    # Attempt to build an inference-mode pipeline when a model path is configured.
    # Fall back to dummy mode gracefully on any loading error.
    inference_active = False
    model_path = os.environ.get("CROWDDNA_MODEL_PATH")
    calibration_config = None
    if enable_calib and state_pts is not None and len(state_pts) == 4 and world_coords is not None:
        try:
            w_pts = [(float(r[0]), float(r[1])) for r in world_coords]
            calibration_config = CalibrationConfig(enabled=True, image_points=state_pts, world_points_m=w_pts)
            calibration_config.validate()
        except Exception as e:
            return None, [], [], f"Calibration Error: {e}"

    try:
        pipeline = CrowdFlowPipeline(model_path=model_path, calibration_config=calibration_config)
        inference_active = model_path is not None
    except (ModelNotFoundError, UnsupportedModelFormatError) as exc:
        logger.warning(
            "Could not load deployment model (%s). Falling back to dummy mode.", exc
        )
        pipeline = CrowdFlowPipeline(model_path=None, calibration_config=calibration_config)

    try:
        result = pipeline.run(video_file)
    except CrowdFlowError as exc:
        logger.error("[Thread %s] Pipeline failed: %s", thread_id, exc)
        logger.info("[Thread %s] process_video() return - CrowdFlowError", thread_id)
        return None, [], [], f"Error: {exc}"
    except Exception as exc:
        logger.exception("[Thread %s] Unexpected error during pipeline run.", thread_id)
        logger.info("[Thread %s] process_video() return - unexpected exception", thread_id)
        return None, [], [], f"Error: Unexpected failure: {exc}"

    if not getattr(result, "output_video_path", None):
        logger.info("[Thread %s] process_video() return - no output video generated", thread_id)
        return None, [], [], "Error: Pipeline did not generate an output video."

    output_path = result.output_video_path

    timeline_data = _timeline_to_dataframe(result.timeline)
    metadata_data = _metadata_to_rows(result.metadata)

    if inference_active:
        backend = result.metadata.get("backend", "TorchScript")
        fmt = result.metadata.get("model_format", "unknown")
        version = result.metadata.get("model_version") or "unversioned"
        status_msg = (
            f"✅ Analysis complete — inference mode · {backend} ({fmt}) · version: {version}."
        )
    else:
        status_msg = "✅ Analysis complete (dummy mode — no risk model loaded)."
    
    logger.info("[Thread %s] process_video() return - success", thread_id)
    return output_path, timeline_data, metadata_data, status_msg


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CrowdFlow DNA — Crowd Risk Analyser") as demo:
    gr.Markdown(
        """
        # CrowdFlow DNA — Crowd Risk Analyser
        Upload a video to run the end-to-end vision pipeline.
        Operating mode (inference or dummy) is determined dynamically by the `CROWDDNA_MODEL_PATH` environment variable.
        """
    )

    state_image_points = gr.State([])
    state_original_frame = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input Video", format="mp4")
            
            with gr.Accordion("Metric Ground-Plane Calibration", open=False):
                gr.Markdown(
                    '''
                    **IMPORTANT**: Perspective correction alone does not establish real-world scale. 
                    World coordinates must be entered using known physical measurements.
                    The calibration system can establish a metric coordinate transformation only when the supplied physical correspondences are genuinely measured.
                    
                    **Instructions:**
                    1. Upload a video to extract the first frame.
                    2. Click exactly 4 points on the ground plane in the image below.
                       *(Recommended ordering: Top-Left, Top-Right, Bottom-Right, Bottom-Left)*
                    3. Enter the corresponding physical coordinates in meters (X, Y).
                       *(Example: 0,0 | 10,0 | 10,5 | 0,5)*
                    '''
                )
                
                calib_frame = gr.Image(label="First Frame (Click to select points)", interactive=False)
                calib_msg = gr.Textbox(label="Selection Status", interactive=False)
                
                world_coords_df = gr.Dataframe(
                    headers=["X (meters)", "Y (meters)"],
                    datatype=["number", "number"],
                    row_count=(4, "fixed"),
                    col_count=(2, "fixed"),
                    label="World Coordinates",
                    type="array",
                    interactive=True
                )
                
                enable_calibration = gr.Checkbox(label="Enable Calibrated Processing", value=False)
                calib_state = gr.Textbox(label="Calibration State", value="UNCALIBRATED", interactive=False)
                calib_validation_msg = gr.Textbox(label="Validation Result", interactive=False)
                calib_preview = gr.Textbox(label="Metric Preview", interactive=False, lines=5)
                validate_btn = gr.Button("Validate Calibration")
            
            run_btn = gr.Button("Run Analysis", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=1):
            output_video = gr.Video(label="Annotated Output", interactive=False, format="mp4")

    with gr.Row():
        timeline_table = gr.DataFrame(
            label="Risk Timeline",
            headers=["Frame", "Region ID", "Risk Label", "Confidence"],
            interactive=False,
        )

    with gr.Row():
        metadata_table = gr.DataFrame(
            label="Video Metadata",
            headers=["Property", "Value"],
            interactive=False,
        )

    input_video.change(
        fn=get_first_frame,
        inputs=[input_video],
        outputs=[state_original_frame, world_coords_df, calib_msg, calib_frame]
    )
    
    calib_frame.select(
        fn=handle_image_click,
        inputs=[state_original_frame, state_image_points],
        outputs=[calib_frame, state_image_points, calib_msg]
    )
    
    validate_btn.click(
        fn=validate_and_preview,
        inputs=[state_image_points, world_coords_df, enable_calibration],
        outputs=[calib_state, calib_validation_msg, calib_preview]
    )

    run_btn.click(
        fn=process_video,
        inputs=[input_video, state_image_points, world_coords_df, enable_calibration],
        outputs=[output_video, timeline_table, metadata_table, status_box],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info("================ STARTUP SEQUENCE ================")
    logger.info(f"PID: {os.getpid()} | __name__ == '__main__'")
    logger.info("Before launch()")
    
    
    try:
        demo.launch(server_name="0.0.0.0", server_port=port)
        logger.info("Immediately after launch()")
        logger.info("After launch returns")
    except Exception as e:
        logger.info("[Thread %s] Gradio launch exception caught", __import__('threading').get_ident())
        logger.exception(f"Gradio launch failed with exception: {e}")
        raise
