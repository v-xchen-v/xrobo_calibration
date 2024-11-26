from xrobo_calibration import calibrate_camera, calibrate_arm

# Example usage
## Mock data
camera_data = {
    'camera_matrix': 'mock_camera_matrix',
    'distortion_coefficients': 'mock_distortion_coefficients',
    'image_points': 'mock_image_points',
    'object_points': 'mock_object_points',
    'image_size': 'mock_image_size',
}

board_data = {
    'board_size': 'mock_board_size',
    'square_size': 'mock_square_size',
    'board_corners': 'mock_board_corners',
}

calibrate_camera(camera_data, board_data, verbose=True)
calibrate_arm(camera_data, board_data, verbose=True)
