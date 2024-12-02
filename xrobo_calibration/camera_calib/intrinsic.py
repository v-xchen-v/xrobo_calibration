from typing import List, Tuple, Dict
import numpy as np
from .file_io import save_parameters, load_parameters
from typing import Optional
import cv2
from .pattern_detection import generate_object_points
from .utils import calculate_reprojection_error



def calibrate_camera_intrinsic(
    image_points: List[np.ndarray], 
    image_size: Tuple[int, int], 
    pattern_size: Optional[Dict[str, float]] = None,
    object_points: Optional[List[np.ndarray]] = None, 
    verbose: bool = False,
) -> Dict:
    """
    Calibrate the intrinsic parameters of the camera.
    
    Supports two modes of input:
    1. image_points, image_size, and pattern_size: Automatically generates object_points.
    2. image_points, object_points, and image_size: Uses provided object_points directly.
    
    Args:
        image_points (List[np.ndarray]): List of 2D points in image coordinates for each calibration image.
        image_size (tuple): Width and height of the images (e.g., (1920, 1080)).
        pattern_size (Optional[Dict[str, float]]): For mode 1, dimensions of the calibration pattern:
                                                   {"rows": int, "cols": int, "square_size": float}.
        object_points (Optional[List[np.ndarray]]): For mode 2, list of 3D points in object coordinates.

    Returns:
        Dict: Intrinsic calibration results, including camera matrix and distortion coefficients.
        Intrinsic Matrix:
        [[fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]]

        Distortion Coefficients:
        [k1, k2, p1, p2, k3]
    """
    # intrinsic_matrix = np.array([[600, 0, image_size[0] / 2],
    #                               [0, 600, image_size[1] / 2],
    #                               [0, 0, 1]])
    # distortion_coeffs = np.array([0.1, -0.05, 0.01, 0.01, 0])
    
    # Ensure image_points are Numpy arrays
    image_points = [np.array(points, dtype=np.float32) for points in image_points]
    
    # Handling Mode 1: Generate object_points from pattern_size
    if object_points is None and pattern_size is not None:
        object_points = generate_object_points(pattern_size, num_images=len(image_points))
    elif object_points is None:
        raise ValueError("Either object_points or pattern_size must be provided.")
    
    # Ensure object_points are Numpy arrays
    object_points = [np.array(points, dtype=np.float32) for points in object_points]
    
    # Perform intrinsic calibration
    ret, intrinsic_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points, 
        imagePoints=image_points, 
        imageSize=image_size, 
        cameraMatrix=None, 
        distCoeffs=None
    )
    
    # mean_error = 0
    # for i in range(len(object_points)):
    #     imgpoints_est, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], intrinsic_matrix, distortion_coeffs)
    #     error = cv2.norm(image_points[i], imgpoints_est, cv2.NORM_L2)/len(imgpoints_est)
    #     if verbose:
    #         print(f"{i} error: {error}")
    #     mean_error += error
    # if verbose:
    #     print( "total error: {}".format(mean_error/len(object_points)) )
    
    calculate_reprojection_error(object_points=object_points,
                                 image_points=image_points,
                                 tvecs=tvecs,
                                 rvecs=rvecs,
                                 mtx=intrinsic_matrix,
                                 dist=distortion_coeffs,
                                 verbose=verbose)
        
    # Check for calibration success
    if not ret:
        raise RuntimeError("Camera calibration failed. Check the input data.")
    
    return {
        "intrinsic_matrix": intrinsic_matrix, 
        "distortion_coeffs": distortion_coeffs,
        "reprojection_error": ret,
    }

def load_intrinsic(filepath, format="json"):
    """
    Load intrinsic calibration data from a file in the specified format.

    Args:
        filepath (str): Path to the input file.
        format (str): Format of the file ("json" or "npy").

    Returns:
        dict: Intrinsic calibration data.
    """
    data = load_parameters(filepath, format)
    return {
        "intrinsic_matrix": data["intrinsic_matrix"],
        "distortion_coeffs": data["distortion_coeffs"]
    }
    
def save_intrinsic(filepath, intrinsic_matrix, distortion_coeffs, reprojection_error, 
                   format="json"):
    """
    Save intrinsic calibration data to a file in the specified format.

    Args:
        filepath (str): Path to the output file.
        intrinsic_matrix (np.ndarray): Intrinsic matrix.
        distortion_coeffs (np.ndarray): Distortion coefficients.
        reprojection_error (float): Reprojection error from calibration.
        format (str): Format to save the file ("json" or "npy").
    """
    data = {
        "intrinsic_matrix": intrinsic_matrix,
        "distortion_coeffs": distortion_coeffs,
        "reprojection_error": reprojection_error
    }
    save_parameters(filepath, data, format)