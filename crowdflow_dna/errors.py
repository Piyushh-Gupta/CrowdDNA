class CrowdFlowError(Exception):
    """Base exception for all CrowdFlow DNA errors."""
    pass

class InvalidVideoFormatError(CrowdFlowError):
    """Raised when an uploaded file is not an MP4 or AVI."""
    pass

class VideoCorruptionError(CrowdFlowError):
    """Raised when OpenCV fails to read the video file."""
    pass

class UploadSizeExceededError(CrowdFlowError):
    """Raised when a video exceeds the maximum size or duration."""
    pass

class ModelInferenceError(CrowdFlowError):
    """Raised when the inference runtime fails during risk classification."""
    pass