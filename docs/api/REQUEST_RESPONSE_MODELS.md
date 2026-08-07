# Request and Response Models

## Request Models

### `InferenceRequest`
- **Description:** Payload for triggering an inference pipeline via HTTP.
- **Type:** `multipart/form-data`
- **Fields:**
  - `file` (File, Required): The video file (MP4/AVI). Max size: 50MB.

### `ModelLoadRequest`
- **Description:** Payload to load a new model into `InferenceRuntime`.
- **Type:** `application/json`
- **Fields:**
  - `model_path` (String, Required): Path or ID of the model.
  - `version` (String, Optional): Specific model version.

## Response Models

### `JobStatusResponse`
- **Description:** Status of an asynchronous pipeline execution.
- **Fields:**
  - `job_id` (String): UUID of the job.
  - `status` (String): "PENDING", "PROCESSING", "COMPLETED", "FAILED".
  - `progress` (Float): Completion percentage.
  - `result` (`InferenceResponse`, Optional): The final result if completed.

### `InferenceResponse`
- **Description:** Maps directly to `PipelineResult`.
- **Fields:**
  - `timeline` (Array of `TimelineEntry`): Frame-by-frame risk predictions.
  - `metadata` (`VideoMetadata`): Video metrics and stream metadata.
  - `output_video_url` (String): URL to download the annotated video.

### `TimelineEntry`
- **Description:** Maps to `TimelineEntry` dataclass.
- **Fields:**
  - `frame_index` (Integer)
  - `predictions` (Array of `RiskPrediction`)

### `RiskPrediction`
- **Description:** Maps to `RiskPrediction` dataclass.
- **Fields:**
  - `region_id` (Integer): Track ID.
  - `label` (String): "Safe", "Congesting", or "Critical".
  - `confidence` (Float): Prediction confidence [0.0 - 1.0].

### `VideoMetadata`
- **Description:** Stream metadata dictionary.
- **Fields:**
  - `fps` (Float)
  - `width` (Integer)
  - `height` (Integer)
  - `frame_count` (Integer)
  - `duration_seconds` (Float)
  - `sample_rate` (Integer)
