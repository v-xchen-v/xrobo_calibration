from typing import List, Tuple, Dict
import numpy as np
from .file_io import save_parameters, load_parameters
import cv2
from typing import Optional
from .pattern_detection import generate_object_points
from .utils import calculate_reprojection_error

def calibrate_camera_extrinsic_solvepnp(
    image_points: List[np.ndarray], 
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    pattern_size: Optional[Dict[str, float]] = None,
    object_points: Optional[List[np.ndarray]] = None
) -> Dict:
    """
    Calculate extrinsic parameters using solvePnP. 
    This function assumes that the intrinsic parameters are known.

    Supports two modes of input:
    1. image_points, pattern_size, intrinsic_matrix, and distortion_coeffs: Automatically generates object_points.
    2. image_points, object_points, intrinsic_matrix, and distortion_coeffs: Uses provided object_points directly.
    
    Args:
        image_points (List[np.ndarray]): List of 2D points in image coordinates.
        intrinsic_matrix (np.ndarray): Intrinsic camera matrix.
        distortion_coeffs (np.ndarray): Distortion coefficients.
        pattern_size (Optional[Dict[str, float]]): For mode 1, dimensions of the calibration pattern:
                                                   {"rows": int, "cols": int, "square_size": float}.
        object_points (Optional[List[np.ndarray]]): For mode 2, list of 3D points in object coordinates

    Returns:
        Dict: Extrinsic calibration results, including rotation and translation vectors.
    """
    # rotation_vector = np.array([0.1, 0.2, 0.3])
    # translation_vector = np.array([0.5, 0.5, 1.0])
    
    # Handling Mode 1: Generate object_points from pattern_size
    if object_points is None and pattern_size is not None:
        object_points = generate_object_points(pattern_size, num_images=len(image_points))
    elif object_points is None:
        raise ValueError("Either object_points or pattern_size must be provided.")
    
    # make sure object_points and image_points are Numpy arrays
    object_points = np.array([np.array(points, dtype=np.float32) for points in object_points])
    image_points = np.array([np.array(points, dtype=np.float32) for points in image_points])
    intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32)
    distortion_coeffs = np.array(distortion_coeffs, dtype=np.float32)
    
    rvecs = []
    tvecs = []
    
    for i, (obj_pts, img_pts) in enumerate(zip(object_points, image_points)):
        # Solve for the pose of the calibration pattern
        
        ## calibrate extrinsic parameters
        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=intrinsic_matrix,
            distCoeffs=distortion_coeffs)
        
        if not success:
            raise ValueError(f"Failed to calibrate extrinsic parameters for image {i + 1}.")
        
        tvecs.append(translation_vector)
        rvecs.append(rotation_vector)

    # calculate mean reprojection error of each image
    # error = 0
    # for i in range(len(object_points)):
    #     image_points_est, _ = cv2.projectPoints(
    #         objectPoints=object_points[i],
    #         rvec=rvecs[i],
    #         tvec=tvecs[i],
    #         cameraMatrix=intrinsic_matrix,
    #         distCoeffs=distortion_coeffs)
    #     error += cv2.norm(image_points[i], image_points_est, cv2.NORM_L2) / len(image_points[i])
    # mean_error = error / len(object_points)
    mean_error = calculate_reprojection_error(
                    object_points=object_points,
                    image_points=image_points,
                    tvecs=tvecs,
                    rvecs=rvecs,
                    mtx=intrinsic_matrix,
                    dist=distortion_coeffs,
                    verbose=False)
    
    results={"rotation_vector": rvecs, "translation_vector": tvecs,
            "reprojection_error": mean_error}
    return results
   

def calibrate_camera_extrinsic(
    image_points: List[np.ndarray], 
    image_size: Tuple[int, int],
    pattern_size: Optional[Dict[str, float]] = None,
    object_points: Optional[List[np.ndarray]] = None,
    intrinsic_matrix: Optional[np.ndarray] = None,
    distortion_coeffs: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate extrinsic parameters using cv2.calibrateCamera.
    If pass in intrinsic parameters, will refine the given intrinsic parameters instead of estimating them from scratch.
    
    Args:
        image_points (np.ndarray): Corresponding 2D points in the image coordinate system.
        image_size (Tuple[int, int]): Size of the image.
        pattern_size (Optional[Dict[str, float]]): For mode 1, dimensions of the calibration pattern:
        object_points (Optional[List[np.ndarray]]): For mode 2, list of 3D points in object coordinates.
        intrinsic_matrix (np.ndarray): Intrinsic camera matrix.
        distortion_coeffs (np.ndarray): Distortion coefficients.
        
    Returns:
        Dict: Extrinsic calibration results, including rotation and translation vectors.
    """
    if object_points is None and pattern_size is not None:
        object_points = generate_object_points(pattern_size, num_images=len(image_points))
    elif object_points is None:
        raise ValueError("Either object_points or pattern_size must be provided.")
    
    # make sure object_points and image_points are Numpy arrays
    object_points = [np.array(points, dtype=np.float32) for points in object_points]
    image_points = [np.array(points, dtype=np.float32) for points in image_points]
    intrinsic_matrix = np.array(intrinsic_matrix, dtype=np.float32)
    distortion_coeffs = np.array(distortion_coeffs, dtype=np.float32)
    
    # perform extrinsic calibration
    reprojection_error, intrinsic_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=intrinsic_matrix,
        distCoeffs=distortion_coeffs)
    
    results = {
        "intrinsic_matrix": intrinsic_matrix, "distortion_coeffs": distortion_coeffs,
        "rotation_vector": rvecs, "translation_vector": tvecs, 
        "reprojection_error": reprojection_error
    }
    return results

def load_extrinsic(filepath, format="json"):
    """
    Load extrinsic calibration data from a file in the specified format.

    Args:
        filepath (str): Path to the input file.
        format (str): Format of the file ("json" or "npy").

    Returns:
        dict: Extrinsic calibration data.
    """
    data = load_parameters(filepath, format)
    return {
        "rotation_vector": data["rotation_vector"],
        "translation_vector": data["translation_vector"]
    }
    
def save_extrinsic(filepath, rotation_vector, translation_vector, format="json"):
    """
    Save extrinsic calibration data to a file in the specified format.

    Args:
        filepath (str): Path to the output file.
        rotation_vector (np.ndarray): Rotation vector.
        translation_vector (np.ndarray): Translation vector.
        format (str): Format to save the file ("json" or "npy").
    """
    data = {
        "rotation_vector": rotation_vector,
        "translation_vector": translation_vector
    }
    save_parameters(filepath, data, format)