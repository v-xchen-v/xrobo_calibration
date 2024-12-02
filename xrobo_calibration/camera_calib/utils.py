import cv2
import numpy as np

def calculate_reprojection_error(
    object_points, image_points, rvecs, tvecs, mtx, dist, verbose=False
):
    total_error = 0
    total_points = 0

    for i in range(len(object_points)):
        # Project 3D points to the 2D plane
        image_points_est, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        # Compute error for each point
        error = cv2.norm(image_points[i], image_points_est, cv2.NORM_L2)
        total_error += error ** 2
        
        if verbose:
            print(f"Image {i + 1} error: {error}")
        total_points += len(object_points[i])

    error = np.sqrt(total_error / total_points)
    if verbose:
        print(f"Total error: {error}")
    return error