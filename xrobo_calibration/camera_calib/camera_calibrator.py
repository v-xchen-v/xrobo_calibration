import numpy as np
from .intrinsic import calibrate_camera_intrinsic
from .extrinsic import calibrate_camera_extrinsic
from ..common.io_utils import save_json, load_json


class CameraCalibrator:
    def __init__(self, image_size):
        """
        Initialize the CameraCalibrator with the image size.

        Args:
            image_size (tuple): Width and height of the image (e.g., (1920, 1080)).
        """
        self.image_size = image_size
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None

    def calibrate_intrinsic(self, image_points, object_points):
        """
        Perform intrinsic camera calibration.

        Args:
            image_points (list): 2D points in image coordinates.
            object_points (list): 3D points in object coordinates.

        Returns:
            dict: Intrinsic calibration results.
        """
        results = calibrate_camera_intrinsic(image_points, object_points, self.image_size)
        self.intrinsic_matrix = results["intrinsic_matrix"]
        self.distortion_coeffs = results["distortion_coeffs"]
        return results

    def calibrate_extrinsic(self, object_points, image_points):
        """
        Perform extrinsic camera calibration.

        Args:
            object_points (list): 3D points in world coordinates.
            image_points (list): 2D points in image coordinates.

        Returns:
            dict: Extrinsic calibration results.
        """
        if self.intrinsic_matrix is None:
            raise ValueError("Intrinsic parameters must be calibrated first.")
        results = calibrate_camera_extrinsic(object_points, image_points, self.intrinsic_matrix)
        self.rotation_vector = results["rotation_vector"]
        self.translation_vector = results["translation_vector"]
        return results

    def save_intrinsic(self, filepath):
        """
        Save intrinsic calibration data to a file.

        Args:
            filepath (str): Filepath to save the data.
        """
        if self.intrinsic_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Intrinsic parameters not calibrated yet.")
        data = {
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }
        save_json(filepath, data)

    def load_intrinsic(self, filepath):
        """
        Load intrinsic calibration data from a file.

        Args:
            filepath (str): Filepath to load the data.

        Returns:
            dict: Intrinsic calibration data.
        """
        data = load_json(filepath)
        self.intrinsic_matrix = np.array(data["intrinsic_matrix"])
        self.distortion_coeffs = np.array(data["distortion_coeffs"])
        return data
