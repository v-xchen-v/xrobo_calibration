"""se calibrate_camera_extrinsic for extrinsic paramet"""

import os, sys
module_path = os.path.abspath(os.path.join("."))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import cv2
from xrobo_calibration.camera_calib.extrinsic import calibrate_camera_extrinsic_solvepnp
from xrobo_calibration.camera_calib.pattern_detection import get_image_points


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
intrinsic_matrix = intrinsic_parameters["intrinsic_matrix"]
distortion_coefficients = intrinsic_parameters["distortion_coeffs"]


images = [
    cv2.imread(f"data/sample_data/camera_calib/data1/chessboard_0{i}.png")
    for i in range(1, 3)
]
pattern_size = {"rows": 8, "cols": 11, "square_size": 0.02}
image_points = get_image_points(images, pattern_size)
extrinsic_results = calibrate_camera_extrinsic_solvepnp(
    image_points, intrinsic_matrix, distortion_coefficients, pattern_size
)
print(extrinsic_results)
