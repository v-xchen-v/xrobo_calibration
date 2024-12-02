import os, sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import cv2
from xrobo_calibration.camera_calib.pattern_detection import get_image_points, visualize_and_save_image_points

# Load images (assuming images are loaded into a list)
images = [cv2.imread(f"data/sample_data/camera_calib/data1/chessboard_0{i}.png") for i in range(1, 6)]

# Define pattern size
pattern_size = {"rows":8, "cols": 11}

# Detect image points
image_points = get_image_points(images, pattern_size)

# Define image size (assume all images have the same resolution)
image_size = (images[0].shape[1], images[0].shape[0])

# Visualize, and save image points if needed
visualize_and_save_image_points(images, image_points, pattern_size, show=True)