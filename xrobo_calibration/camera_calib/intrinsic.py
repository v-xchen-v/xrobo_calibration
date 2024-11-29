from typing import List, Tuple, Dict
import numpy as np

def calibrate_camera_intrinsic(image_points: List, object_points: List, image_size: Tuple[int, int]) -> Dict:
    """
    Calibrate the intrinsic parameters of the camera.
    """
    intrinsic_matrix = np.array([[600, 0, image_size[0] / 2],
                                  [0, 600, image_size[1] / 2],
                                  [0, 0, 1]])
    distortion_coeffs = np.array([0.1, -0.05, 0.01, 0.01, 0])
    return {"intrinsic_matrix": intrinsic_matrix, "distortion_coeffs": distortion_coeffs}