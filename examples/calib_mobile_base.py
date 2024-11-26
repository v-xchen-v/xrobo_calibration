from xrobo_calibration import calibrate_camera, calibrate_mobile_base, calibrate_arm

# Mock Example data
camera_data = {...}
arm_data = {...}
mobile_base_data = {"sensor_offsets": [0.1, -0.1]}
odometry_data = [0.1, 0.15, 0.2]

# Perform calibrations
camera_result = calibrate_camera(camera_data)
arm_result = calibrate_arm(arm_data)
base_result = calibrate_mobile_base(mobile_base_data, odometry_data)

print("Camera Calibration:", camera_result)
print("Arm Calibration:", arm_result)
print("Mobile Base Calibration:", base_result)