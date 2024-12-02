import os, sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


import cv2
from xrobo_calibration.camera_calib import CameraCalibrator


# Detect image points and visualize
images = [cv2.imread(f"data/sample_data/camera_calib/data1/chessboard_0{i}.png") for i in range(1, 6)]

# Define image size (assume all images have the same resolution)
image_size = (images[0].shape[1], images[0].shape[0])
print(f'Image Size:\n{image_size}')
# Initialize calibrator
camera_calibrator = CameraCalibrator(image_size=image_size)

pattern_size = {"rows": 8, "cols": 11, "square_size": 0.02}
image_points = camera_calibrator.get_image_points(images, pattern_size, show=False)

# Assume intrinsic parameters are known
intrinsic_parameters = {
    "intrinsic_matrix": [
        [639.6522328966562, 0.0, 641.5064383916216],
        [0.0, 639.7681175968406, 362.99104548918365],
        [0.0, 0.0, 1.0],
    ],
    "distortion_coeffs": [
        [
            -0.051020305438604675,
            0.060096711285673926,
            -0.0009498526881965997,
            -0.0002917577737097018,
            -0.021150451371843158,
        ]
    ],
    "reprojection_error": 0.05935032059833209,
}
camera_calibrator.intrinsic_matrix = intrinsic_parameters["intrinsic_matrix"]
camera_calibrator.distortion_coeffs = intrinsic_parameters["distortion_coeffs"]

extrinsic_results = camera_calibrator.calibrate_extrinsic_single_image(
    image_points=[image_points[0]],
    pattern_size=pattern_size,
)

print('Extrinsic Parameters:')
print(extrinsic_results)