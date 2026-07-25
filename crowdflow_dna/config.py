"""
CrowdFlow DNA — Inference Pipeline Configuration

Architectural Note:
This configuration module governs the inference pipeline, which operates in
2D image space (pixels). Constants like PROXIMITY_RADIUS are in pixel units
because the pipeline lacks camera calibration.

This is intentionally separate from `configs/default.yaml`, which governs
the training simulation space where coordinates are physical (metres).
Do not collapse this boundary.
"""

# Video Ingestion
MAX_FILE_SIZE_MB = 200
MAX_DURATION_SECONDS = 300
FRAME_SAMPLE_RATE = 5  # Process every 5th frame to meet latency targets

# Detection & Tracking
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Graph Construction
PROXIMITY_RADIUS = 50.0  # Pixels (distance threshold for edges)
GRID_ROWS = 10
GRID_COLS = 10

# Performance & Scalability Settings
PERFORMANCE_MAX_BATCH_SIZE = 32
PERFORMANCE_BATCH_TIMEOUT_MS = 50
PERFORMANCE_CACHE_MAX_SIZE = 1000
PERFORMANCE_CACHE_TTL_SECONDS = 300
PERFORMANCE_MAX_THREADS = 10
PERFORMANCE_MAX_PROCESSES = 4
PERFORMANCE_PROFILER_SAMPLING_RATE = 1.0