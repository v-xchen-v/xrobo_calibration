from typing import List, Dict, Tuple
import cv2
import numpy as np
import os

def get_image_points(
    images: List[np.ndarray], 
    pattern_size: Dict[str, int], 
    show: bool = False,
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

    if show:
        visualize_and_save_image_points(images, image_points, pattern_size, show=True)
        
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

def generate_object_points(
    pattern_size: Dict[str, int], 
    num_images: int,
) -> np.ndarray:
    """
    Generate object points for a calibration pattern based on the pattern size and square size.

    Args:
        pattern_size (Dict[str, int]): Dimensions of the calibration pattern {"rows": int, "cols": int}.
        num_images (int): Number of images for which object points are generated.
        
    Returns:
        np.ndarray: 3D object points for the calibration pattern.
    """
    rows, cols, square_size = pattern_size["rows"], pattern_size["cols"], pattern_size["square_size"]
    
    # Generate object points for a single pattern
    object_points_single = np.zeros((rows * cols, 3), np.float32)
    object_points_single[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )
    # Replicate the object points for all images
    object_points = [object_points_single for _ in range(num_images)]
    
    return object_points