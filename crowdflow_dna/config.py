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