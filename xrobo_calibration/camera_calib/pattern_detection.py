from typing import List, Dict, Tuple
import cv2
import numpy as np
import os

def get_image_points(
    images: List[np.ndarray], 
    pattern_size: Dict[str, int], 
    criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
) -> List[np.ndarray]:
    """
    Detect and extract 2D image points from input images using the given calibration pattern.

    Args:
        images (List[np.ndarray]): List of input images (in grayscale or BGR).
        pattern_size (Dict[str, int]): Dimensions of the calibration pattern:
                                       {"rows": int, "cols": int}.
        criteria (Tuple[int, int, float]): Criteria for corner refinement.

    Returns:
        List[np.ndarray]: List of 2D image points detected in each input image.
    """
    rows = pattern_size["rows"]
    cols = pattern_size["cols"]

    image_points = []  # To store points for all images

    for i, img in enumerate(images):
        # Convert image to grayscale if needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            # Refine corner locations for better accuracy
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            image_points.append(corners)
        else:
            print(f"Pattern not detected in image {i + 1}. Skipping this image.")

    if not image_points:
        raise RuntimeError("No valid image points detected. Check the input images and pattern size.")

    return image_points

def visualize_and_save_image_points(
    images: List[np.ndarray],
    image_points: List[np.ndarray],
    pattern_size: Dict[str, int],
    output_dir: str = None,
    show: bool = False
) -> None:
    """
    Visualize or save images with detected image points.

    Args:
        images (List[np.ndarray]): List of input images (grayscale or BGR).
        image_points (List[np.ndarray]): List of 2D points detected in each image.
        pattern_size (Dict[str, int]): Dimensions of the calibration pattern {"rows": int, "cols": int}.
        output_dir (str): Directory to save the images with visualized points. If None, no images are saved.
        show (bool): Whether to display the images with visualized points.

    Raises:
        ValueError: If `images` and `image_points` lengths do not match.
    """
    if len(images) != len(image_points):
        raise ValueError("Number of images and image points must match.")

    rows = pattern_size["rows"]
    cols = pattern_size["cols"]

    # Ensure output directory exists if saving is enabled
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, (img, points) in enumerate(zip(images, image_points)):
        # Draw the detected points on the image
        img_with_points = img.copy()
        cv2.drawChessboardCorners(img_with_points, (cols, rows), points, True)

        # Show the image if enabled
        if show:
            cv2.imshow(f"Image {i + 1}", img_with_points)
            cv2.waitKey(0)

        # Save the image if output directory is specified
        if output_dir:
            output_path = os.path.join(output_dir, f"image_with_points_{i + 1}.jpg")
            cv2.imwrite(output_path, img_with_points)

    if show:
        cv2.destroyAllWindows()
