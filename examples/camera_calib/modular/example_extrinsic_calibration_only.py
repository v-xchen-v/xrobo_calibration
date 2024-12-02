"""se calibrate_camera_extrinsic for extrinsic paramet"""

from xrobo_calibration.camera_calib import calibrate_camera_extrinsic

extrinsic_results = calibrate_camera_extrinsic(object_points, image_points, intrinsic_matrix)
print(extrinsic_results)