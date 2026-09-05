import cv2
import numpy as np

def _check_polygon_validity(pts: np.ndarray, name: str, min_area: float = 1.0) -> None:
    # Check for duplicates or near-duplicates
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-3:
                raise ValueError(f"{name} has duplicate or near-duplicate points.")
                
    # Check for collinearity / zero area / self-intersection using cv2.contourArea
    # cv2.isContourConvex checks for convexity (which prevents self-intersecting / bowties)
    pts_int = np.round(pts).astype(np.int32)
    # contourArea might be 0 for collinear points
    area = cv2.contourArea(pts_int)
    if area < min_area:
        raise ValueError(f"{name} does not form a valid quadrilateral with non-negligible area.")
        
    # Check for self-intersection / convexity
    # If the points form a convex polygon, they cannot be self-intersecting (bowtie).
    if not cv2.isContourConvex(pts_int):
        raise ValueError(f"{name} quadrilateral is self-intersecting or non-convex.")
