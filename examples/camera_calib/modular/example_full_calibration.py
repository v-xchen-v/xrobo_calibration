"""Use calibrate_camera_intrinsic for intrinsic parameters."""

import os, sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cv2
from xrobo_calibration.camera_calib.pattern_detection import get_image_points
from xrobo_calibration.camera_calib.intrinsic import calibrate_camera_intrinsic
from xrobo_calibration.camera_calib.extrinsic import calibrate_camera_extrinsic

# Load images (assuming images are loaded into a list)
images = [cv2.imread(f"data/sample_data/camera_calib/data1/chessboard_0{i}.png") for i in range(1, 6)]

# Define pattern size
pattern_size = {"rows":8, "cols": 11, "square_size": 0.02}

# Detect image points
image_points = get_image_points(images, pattern_size)

# Define image size (assume all images have the same resolution)
image_size = (images[0].shape[1], images[0].shape[0])

# Do calibration
intrinsic_results = calibrate_camera_intrinsic(image_points, image_size, pattern_size)
print(intrinsic_results)


extrinsic_results = calibrate_camera_extrinsic(image_points=image_points,
                           image_size=image_size,
                           pattern_size=pattern_size,
                           intrinsic_matrix=intrinsic_results['intrinsic_matrix'],
                           distortion_coeffs=intrinsic_results['distortion_coeffs'])
print(extrinsic_results)