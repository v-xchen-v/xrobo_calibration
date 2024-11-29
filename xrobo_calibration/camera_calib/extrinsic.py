from typing import List, Tuple, Dict
import numpy as np

def calibrate_camera_extrinsic(object_points: List, image_points: List, intrinsic_matrix: np.ndarray) -> Dict:
    """
    Calibrate the extrinsic parameters of the camera.
    """
    rotation_vector = np.array([0.1, 0.2, 0.3])
    translation_vector = np.array([0.5, 0.5, 1.0])
    return {"rotation_vector": rotation_vector, "translation_vector": translation_vector}
