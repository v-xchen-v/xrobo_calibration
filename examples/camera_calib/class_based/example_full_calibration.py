import os, sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


import cv2
from xrobo_calibration.camera_calib import CameraCalibrator
from xrobo_calibration.camera_calib.pattern_detection import generate_object_points

# Detect image points and visualize
images = [cv2.imread(f"data/sample_data/camera_calib/data1/chessboard_0{i}.png") for i in range(1, 6)]

# Define image size (assume all images have the same resolution)
image_size = (images[0].shape[1], images[0].shape[0])
print(f'Image Size:\n{image_size}')
# Initialize calibrator
camera_calibrator = CameraCalibrator(image_size=image_size)

pattern_size = {"rows": 8, "cols": 11, "square_size": 0.02}
image_points = camera_calibrator.get_image_points(images, pattern_size, show=False)

camera_calibrator.calibrate_intrinsic(image_points, pattern_size=pattern_size)

object_points = generate_object_points(
    pattern_size, num_images=len(image_points))

extrinsic_results = camera_calibrator.calibrate_extrinsic_per_image(
    image_points=image_points,
    object_points=object_points,
    method="calibrateCamera",
)

print('Intrinsic Parameters:')
print('Intrinsic Matrix:')
print(camera_calibrator.intrinsic_matrix)
print('Distortion Coefficients:')
print(camera_calibrator.distortion_coeffs)
print('Extrinsic Parameters:')
print(extrinsic_results)

# Visualize and save the projected points
camera_calibrator.visualize_projected_corners(
    images=images,
    object_points=object_points,
    output_dir="data/sample_data/camera_calib/output",
    pattern_size=(pattern_size["cols"], pattern_size["rows"]),
    show=False
)