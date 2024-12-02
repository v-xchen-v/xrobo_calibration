import cv2
import numpy as np
from typing import List


def visualize_and_save_projected_corners(
    images: List[np.ndarray],
    object_points: List[np.ndarray],
    rotation_vectors: List[np.ndarray],
    translation_vectors: List[np.ndarray],
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    pattern_size: tuple,
    output_dir: str = None,
    show: bool = False
) -> None:
    """
    Visualize and optionally save the projected 3D points on 2D images using cv2.drawChessboardCorners.

    Args:
        images (List[np.ndarray]): List of input images (grayscale or BGR).
        object_points (List[np.ndarray]): List of 3D points for each image.
        rotation_vectors (List[np.ndarray]): List of rotation vectors for each image.
        translation_vectors (List[np.ndarray]): List of translation vectors for each image.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
        distortion_coeffs (np.ndarray): Lens distortion coefficients.
        pattern_size (tuple): Chessboard pattern size as (rows, cols).
        output_dir (str): Directory to save the images with projected corners. If None, no images are saved.
        show (bool): Whether to display the images with visualized points.

    Raises:
        ValueError: If the lengths of the input lists do not match.
    """
    if len(images) != len(object_points) or len(object_points) != len(rotation_vectors):
        raise ValueError("Mismatch in the number of images, object_points, or extrinsic parameters.")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    for i, (img, obj_pts, rvec, tvec) in enumerate(zip(images, object_points, rotation_vectors, translation_vectors)):
        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(
            objectPoints=obj_pts,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=intrinsic_matrix,
            distCoeffs=distortion_coeffs
        )

        # Convert projected points into required shape for cv2.drawChessboardCorners
        corners = projected_points.reshape(-1, 1, 2).astype(np.float32)

        # Draw chessboard corners on the image
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(
            image=img_with_corners,
            patternSize=pattern_size,
            corners=corners,
            patternWasFound=True  # Assume the pattern was detected correctly
        )

        # Show the image if enabled
        if show:
            cv2.imshow(f"Image {i + 1}", img_with_corners)
            cv2.waitKey(0)

        # Save the image if output directory is specified
        if output_dir:
            save_path = f"{output_dir}/image_with_chessboard_corners_{i + 1}.jpg"
            cv2.imwrite(save_path, img_with_corners)

    if show:
        cv2.destroyAllWindows()
