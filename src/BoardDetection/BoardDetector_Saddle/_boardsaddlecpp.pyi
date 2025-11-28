"""
Type stubs for chessboard_detector module

This module provides chessboard corner detection functionality
for computer vision applications.
"""

import numpy as np
from typing import Optional, Tuple
import numpy.typing as npt


class DetectionConfig:
    """
    Configuration parameters for chessboard detection.

    Attributes:
        min_pts_needed: Minimum number of valid grid points required (default: 15)
        max_pts_needed: Maximum number of points to search for before stopping (default: 25)
        max_px_dist: Maximum pixel distance for matching grid to saddle points (default: 5.0)
        square_eps: Tolerance for square shape validation (default: 3.0)
        saddle_window_size: Window size for saddle point refinement (default: 4)
        max_image_size: Maximum dimension for image resizing (default: 500)
        gradient_threshold_multiplier: Multiplier for gradient masking threshold (default: 2.0)
        canny_low: Lower threshold for Canny edge detection (default: 20)
        canny_high: Upper threshold for Canny edge detection (default: 250)
    """

    min_pts_needed: int
    max_pts_needed: int
    max_px_dist: float
    square_eps: float
    saddle_window_size: int
    max_image_size: int
    gradient_threshold_multiplier: float
    canny_low: float
    canny_high: float

    def __init__(self) -> None:
        """Initialize with default configuration values."""
        ...


def detect_corners(
    image: npt.NDArray[np.uint8],
    config: Optional[DetectionConfig] = None
) -> npt.NDArray[np.float32]:
    """
    Detect chessboard corners in an image.

    This function identifies the four corner points of a chessboard in an image
    using saddle point detection and perspective transform fitting.

    Args:
        image: Input image as numpy array with shape (H, W, C) where C is 1, 3, or 4.
               RGB/RGBA images will be converted to grayscale internally.
        config: Optional configuration object. If None, uses default settings.

    Returns:
        Array of shape (4, 2) containing the (x, y) coordinates of the four
        chessboard corners in counter-clockwise order starting from top-left:
        [[x0, y0],   # top-left
         [x1, y1],   # top-right
         [x2, y2],   # bottom-right
         [x3, y3]]   # bottom-left

    Raises:
        RuntimeError: If chessboard detection fails or input image is invalid.
        ValueError: If image has wrong number of dimensions or unsupported format.

    Example:
        >>> import numpy as np
        >>> import chessboard_detector as cd
        >>> 
        >>> # Load your image
        >>> image = np.array(...)  # shape (H, W, 3)
        >>> 
        >>> # Detect corners with default settings
        >>> corners = cd.detect_corners(image)
        >>> 
        >>> # Or with custom configuration
        >>> config = cd.DetectionConfig()
        >>> config.min_pts_needed = 20
        >>> config.max_px_dist = 10.0
        >>> corners = cd.detect_corners(image, config)
    """
    ...
