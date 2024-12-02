# Usage Example

This package documents several simple use cases for you to try out. For full details, see the api and class refence.

## Camera calibration
```python
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

camera_calibrator.calibrate_intrinsic(image_points, pattern_size=pattern_size)

extrinsic_results = camera_calibrator.calibrate_extrinsic_per_image(
    image_points=image_points,
    pattern_size=pattern_size,
    method="calibrateCamera",
)
```