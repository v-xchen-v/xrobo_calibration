"""Use calibrate_camera_intrinsic for intrinsic parameters."""

from xrobo_calibration.camera_calib import calibrate_camera_intrinsic

intrinsic_results = calibrate_camera_intrinsic(image_points, object_points, image_size)
print(intrinsic_results)