"""
Geometry utilities for coordinate transformations.

Handles rotation, perspective projection, and matrix applications for
ensuring OCR annotations match image transformations.
"""

import math
import numpy as np
from typing import List, Tuple, Union

def rotate_point(
    point: Tuple[float, float],
    center: Tuple[float, float],
    angle_degrees: float
) -> Tuple[float, float]:
    """
    Rotate a point around a center.

    Args:
        point: (x, y) coordinates
        center: (cx, cy) coordinates of rotation center
        angle_degrees: Rotation angle in degrees (negative for clockwise)

    Returns:
        (x, y) new coordinates
    """
    x, y = point
    cx, cy = center

    # Convert to radians
    # Negate angle because in image coordinates (y down), standard rotation matrix
    # rotates clockwise if we don't negate. PIL rotates CCW for positive angles.
    angle_rad = math.radians(-angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Translate to origin, rotate, translate back
    x_rel = x - cx
    y_rel = y - cy

    x_new = x_rel * cos_a - y_rel * sin_a + cx
    y_new = x_rel * sin_a + y_rel * cos_a + cy

    return x_new, y_new

def apply_perspective_transform(
    point: Tuple[float, float],
    matrix: Union[List[List[float]], np.ndarray]
) -> Tuple[float, float]:
    """
    Apply a 3x3 perspective transformation matrix to a 2D point.

    Args:
        point: (x, y) coordinates
        matrix: 3x3 homography matrix (list or numpy array)

    Returns:
        (x, y) new coordinates
    """
    x, y = point

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    # Homogeneous coordinates
    pt = np.array([x, y, 1.0])

    # Apply matrix
    transformed = matrix @ pt

    # Perspective division
    z = transformed[2]
    if abs(z) > 1e-6:
        x_new = transformed[0] / z
        y_new = transformed[1] / z
    else:
        x_new = transformed[0]
        y_new = transformed[1]

    return x_new, y_new
