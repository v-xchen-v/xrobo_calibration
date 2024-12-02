"""Use calibrate_camera for full calibration."""

from xrobo_calibration.camera_calib import calibrate_camera

# Example data
image_points = [...]  # 2D points
object_points = [...]  # 3D points
image_size = (1920, 1080)

calibration_results = calibrate_camera(image_points, object_points, image_size)
print(calibration_results)