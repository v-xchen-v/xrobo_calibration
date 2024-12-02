import numpy as np
from .intrinsic import calibrate_camera_intrinsic
from .extrinsic import calibrate_camera_extrinsic_solvepnp, calibrate_camera_extrinsic
from .pattern_detection import get_image_points
from ..common.io_utils import save_json, load_json
import cv2
import json
import os
import datetime
from typing import List, Optional, Dict
from .pattern_detection import generate_object_points
from .visualization import visualize_and_save_projected_corners

class CameraCalibrator:
    def __init__(self, image_size, log_file="calibration_log.json"):
        """
        Initialize the CameraCalibrator with the image size.

        Args:
            image_size (tuple): Width and height of the image (e.g., (1920, 1080)).
        """
        self.image_size = image_size
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.rotation_vectors = None
        self.translation_vectors = None
        self.log_file = log_file

    def get_image_points(self, images: list, pattern_size: dict, show=False):
        """
        Detect image points in a set of images.

        Args:
            images (list): List of input images.
            pattern_size (dict): Pattern size (rows, cols, square_size).
            show (bool): Whether to display the detected points.

        Returns:
            list: Detected image points for each image.
        """
        image_points = get_image_points(images, pattern_size, show=show)
        return image_points
    
    def calibrate_intrinsic(self, image_points, 
                            object_points: Optional[List[np.ndarray]] = None, 
                            pattern_size: Optional[Dict[str, float]] = None):
        """
        Perform intrinsic camera calibration.
        
        Supports two modes of input:
        1. image_points, image_size, and pattern_size: Automatically generates object_points.
        2. image_points, object_points, and image_size: Uses provided object_points directly.
    
        Args:
            image_points (list): 2D points in image coordinates.
            pattern_size (Optional[Dict[str, float]]): For mode 1, dimensions of the calibration pattern:
                                                    {"rows": int, "cols": int, "square_size": float}.
            object_points (Optional[List[np.ndarray]]): For mode 2, list of 3D points in object coordinates.


        Returns:
            dict: Intrinsic calibration results. format:
            {"intrinsic_matrix": np.ndarray, "distortion_coeffs": np.ndarray, "projection_error": float}

            Intrinsic Matrix:
            [[fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]]

            Distortion Coefficients:
            [k1, k2, p1, p2, k3]
            
            Projection Error: 
            Mean reprojection error.
        """
        results = calibrate_camera_intrinsic(
            image_points=image_points, 
            image_size=self.image_size,
            object_points=object_points, 
            pattern_size=pattern_size)
        
        self.intrinsic_matrix = results["intrinsic_matrix"]
        self.distortion_coeffs = results["distortion_coeffs"]
        return results
    
    def calibrate_extrinsic_single_image(
        self,
        image_points: np.ndarray,
        pattern_size: Optional[Dict[str, float]] = None,
        object_points: Optional[List[np.ndarray]] = None, 
    ) -> Dict[str, np.ndarray]:
        """
        Calibrate extrinsic parameters for a single image.

        Supports two modes of input:
        1. image_points, image_size, and pattern_size: Automatically generates object_points.
        2. image_points, object_points, and image_size: Uses provided object_points directly.
        
        Args:
            object_points (np.ndarray): 3D points in the world coordinate system.
            image_points (np.ndarray): Corresponding 2D points in the image coordinate system.

        Returns:
            Dict: Extrinsic parameters including rotation and translation vectors.
        """
        if self.intrinsic_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Intrinsic parameters must be calibrated first or provided.")

                # Handling Mode 1: Generate object_points from pattern_size
        if object_points is None and pattern_size is not None:
            object_points = generate_object_points(pattern_size, num_images=len(image_points))
        elif object_points is None:
            raise ValueError("Either object_points or pattern_size must be provided.")
        
        # Ensure object_points are Numpy arrays
        object_points = [np.array(points, dtype=np.float32) for points in object_points]
    
        # Calculate pose for the single image
        results = calibrate_camera_extrinsic_solvepnp(
            object_points=object_points,
            image_points=image_points,
            intrinsic_matrix=self.intrinsic_matrix,
            distortion_coeffs=self.distortion_coeffs
        )

        self.rotation_vectors = results["rotation_vector"]
        self.translation_vectors = results["translation_vector"]
        
        return {
            "rotation_vector": results["rotation_vector"],
            "translation_vector": results["translation_vector"],
            "reprojection_error": results["reprojection_error"]
        }
    
    def calibrate_extrinsic_per_image(self, 
                            image_points: List[np.ndarray], 
                            pattern_size: Optional[Dict[str, float]] = None,
                            object_points: Optional[List[np.ndarray]] = None, 
                            method: str ="calibrateCamera"):
        """
        Perform extrinsic camera calibration.
        
        Supports two modes of input:
        1. image_points, image_size, and pattern_size: Automatically generates object_points.
        2. image_points, object_points, and image_size: Uses provided object_points directly.
        
        Supports two methods:
        1. "solvePnP": Use cv2.solvePnP for extrinsic calibration. Assumes intrinsic parameters are already calibrated.
        2. "calibrateCamera": Use cv2.calibrateCamera for extrinsic calibration with unknown intrinsic parameters.

        Args:
            image_points (list): 2D points in image coordinates.
            pattern_size (Optional[Dict[str, float]]): For mode 1, dimensions of the calibration pattern:
                                                   {"rows": int, "cols": int, "square_size": float}.
            object_points (Optional[List[np.ndarray]]): For mode 2, list of 3D points in object coordinates.

            method (str): Method to use for extrinsic calibration ("solvePnP" or "calibrateCamera").

        Returns:
            list: Extrinsic calibration results for each image. Each item is a dict with keys:
            {"rotation_vector": np.ndarray, "translation_vector": np.ndarray, "reprojection_error": float}
        """
        
        # Handling Mode 1: Generate object_points from pattern_size
        if object_points is None and pattern_size is not None:
            object_points = generate_object_points(pattern_size, num_images=len(image_points))
        elif object_points is None:
            raise ValueError("Either object_points or pattern_size must be provided.")
        
        # Ensure object_points are Numpy arrays
        object_points = [np.array(points, dtype=np.float32) for points in object_points]
        
        if len(object_points) != len(image_points):
            raise ValueError("The number of object_points and image_points must match.")
        
        extrinsic_params = []
        
        if method == "solvePnP":
            if self.intrinsic_matrix is None or self.distortion_coeffs is None:
                raise ValueError("Intrinsic parameters must be calibrated first.")
            
            # Perform extrinsic calibration using cv2.solvePnP
            results = calibrate_camera_extrinsic_solvepnp(
                object_points=object_points, 
                image_points=image_points, 
                intrinsic_matrix=self.intrinsic_matrix, 
                distortion_coeffs=self.distortion_coeffs)
            
            # Loop through each pair
            for i, (rvec, tvec) in enumerate(zip(results["rotation_vector"], results["translation_vector"])):
                extrinsic_params.append({
                    "rotation_vector": rvec,
                    "translation_vector": tvec,
                    "reprojection_error": results["reprojection_error"]
                })
            
        elif method == "calibrateCamera":
            # Use cv2.calibrateCamera for intrinsic and extrinsic calibration
            results = calibrate_camera_extrinsic(
                object_points=object_points, 
                image_points=image_points, 
                image_size=self.image_size)
            
            # Loop through each pair
            for i, (rvec, tvec) in enumerate(zip(results["rotation_vector"], results["translation_vector"])):
                extrinsic_params.append({
                    "rotation_vector": rvec,
                    "translation_vector": tvec,
                    "reprojection_error": results["reprojection_error"]
                })
                
            self.intrinsic_matrix = results["intrinsic_matrix"]
            self.distortion_coeffs = results["distortion_coeffs"]
            
        total_mean_error = np.mean([params["reprojection_error"] for params in extrinsic_params])
        results = {
            "rotation_vector": [params["rotation_vector"] for params in extrinsic_params],
            "translation_vector": [params["translation_vector"] for params in extrinsic_params],
            "reprojection_error": total_mean_error
        }

        self.rotation_vectors = results["rotation_vector"]
        self.translation_vectors = results["translation_vector"]
        
        return results

    def visualize_projected_corners(
        self,
        images: List[np.ndarray],
        object_points: List[np.ndarray],
        pattern_size: tuple,
        output_dir: str = None,
        show: bool = False
    ) -> None:
        """
        Visualize and save projected 3D points as chessboard corners on the images.

        Args:
            images (List[np.ndarray]): List of input images.
            object_points (List[np.ndarray]): List of 3D object points for each image.
            pattern_size (tuple): Chessboard pattern size as (rows, cols).
            output_dir (str): Directory to save the images with projected points.
            show (bool): Whether to display the images with points.
        """
        if not self.rotation_vectors or not self.translation_vectors:
            raise ValueError("Extrinsic parameters are not calibrated yet.")

        visualize_and_save_projected_corners(
            images=images,
            object_points=object_points,
            rotation_vectors=self.rotation_vectors,
            translation_vectors=self.translation_vectors,
            intrinsic_matrix=self.intrinsic_matrix,
            distortion_coeffs=self.distortion_coeffs,
            pattern_size=pattern_size,
            output_dir=output_dir,
            show=show
        )

        
    def log_result(self, data):
        """
        Log calibration results and error metrics to a file.

        Args:
            data (dict): Data to log, including calibration parameters and errors.
        """
        data["timestamp"] = datetime.now().isoformat()
        if os.path.exists(self.log_file):
            with open(self.log_file, "r+") as f:
                log_data = json.load(f)
                log_data.append(data)
                f.seek(0)
                json.dump(log_data, f, indent=4)
        else:
            with open(self.log_file, "w") as f:
                json.dump([data], f, indent=4)
                
    def save_intrinsic(self, filepath):
        """
        Save intrinsic calibration data to a file.

        Args:
            filepath (str): Filepath to save the data.
            format (str): File format ("json" or "npy", default: "json").
        """
        if self.intrinsic_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Intrinsic parameters not calibrated yet.")
        data = {
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }
        if format == "json":
            with open(filepath, "w") as f:
                json.dump({k: v.tolist() for k, v in data.items()}, f, indent=4)
        elif format == "npy":
            np.save(filepath, data)
        else:
            raise ValueError("Invalid format. Use 'json' or 'npy'.")    

    def load_intrinsic(self, filepath, format="json"):
        """
        Load intrinsic calibration data from a file.

        Args:
            filepath (str): Filepath to load the data.
            format (str): File format ("json" or "npy", default: "json").

        Returns:
            dict: Intrinsic calibration data.
        """
        if format == "json":
            data = load_json(filepath)
            self.intrinsic_matrix = np.array(data["intrinsic_matrix"])
            self.distortion_coeffs = np.array(data["distortion_coeffs"])
        elif format == "npy":
            data = np.load(filepath)
            self.intrinsic_matrix = np.array(data["intrinsic_matrix"])
            self.distortion_coeffs = np.array(data["distortion_coeffs"])
        else:
            raise ValueError("Invalid format. Use 'json' or 'npy'.")
        
        return data

    def calculate_reprojection_error(self, object_points, image_points):
        """
        Calculate reprojection error for the current calibration parameters.

        Args:
            object_points (list): 3D points in the world coordinate system.
            image_points (list): Corresponding 2D points in the image.

        Returns:
            float: Mean reprojection error.
        """
        if self.intrinsic_matrix is None or self.rotation_vectors is None or self.translation_vectors is None:
            raise ValueError("Calibration parameters are incomplete. Perform calibration first.")

        # Project 3D points to 2D using the calibrated parameters
        projected_points, _ = cv2.projectPoints(
            np.array(object_points),
            self.rotation_vectors,
            self.translation_vectors,
            self.intrinsic_matrix,
            self.distortion_coeffs
        )

        # Compute error
        projected_points = projected_points.squeeze()  # Remove unnecessary dimensions
        image_points = np.array(image_points)
        error = np.sqrt(np.sum((projected_points - image_points) ** 2, axis=1)).mean()

        return error
    
    def validate_calibration(self, validation_object_points, validation_image_points):
        """
        Validate the calibration using independent validation data.

        Args:
            validation_object_points (list): 3D points in the validation dataset.
            validation_image_points (list): 2D points in the validation dataset.

        Returns:
            dict: Validation metrics including reprojection error.
        """
        # Calculate reprojection error on the validation dataset
        validation_error = self.calculate_reprojection_error(validation_object_points, validation_image_points)

        return {"validation_reprojection_error": validation_error}