# xrobo_calibration
The xrobo_calibration project provides a robust framework for calibrating robotic systems. It supports tools and methodologies for determining transformations between key components, such as cameras, robotic arms, and mobile base using calibration patterns like chessboards or markers. 

## Features
- Camera calibration (`xrobot_camera_calib`)
- Robotic arm calibration (`xrobot_arm_calib`)
- Mobile base calibration (`xrobot_mobile_base_calib`)

## Installation
```bash
pip install xrobo_calibration
```

## Usage
```python
from xrobo_calibration import calibrate

calibrate(robot_model, sensor_data)
```
## License
This project is licensed under the MIT License.

## Tutorials and Visualizations

The `notebooks/` directory contains Jupyter notebooks for tutorials and visualizations. These notebooks demonstrate how to use the package and provide tools to visualize calibration results.

## Advanced Tutorials

In addition to the basic package tutorials, the `notebooks/detailed_tutorials/` folder contains advanced topics, including:

- **3D Transformation Basics**: Understand the mathematics and visualization of transformations.
- **Kinematics Primer**: Learn forward and inverse kinematics for robotic arms.
- **Calibration Algorithms**: Dive into advanced calibration techniques like least squares.

### Running Tutorials
1. Install JupyterLab:
   ```bash
   pip install jupyterlab

2. Launch the notebooks
```bash
jupyter-lab notebooks/detailed_tutorials/
```
---

### **Benefits**
- Keeps basic tutorials clean and focused on package usage.
- Provides a dedicated space for users interested in deeper learning or advanced techniques.
- Offers a foundation for community contributions to advanced topics.

Let me know if you need help creating any specific tutorial content!


### Running Notebooks
1. Install JupyterLab:
   ```bash
   pip install jupyterlab
2. Launch the notebooks:
    ```bash
    jupyter-lab notebooks/


## Sample Data

The `data/` folder contains sample images, point clouds, and configuration files for use with tutorials and visualizations.

### Contents
- `images/`: Sample images for camera calibration (e.g., chessboard images).
- `point_clouds/`: Example `.pcd` files for 3D object alignment.
- `configs/`: Configuration files for cameras, robotic arms, and mobile bases.
- `mobile_base/`: Odometry logs and wheel configuration files for mobile base calibration.

### Accessing the Data
You can load the data programmatically using helper functions:
```python
from xrobot_calibration.common.data_utils import get_sample_data_path
path = get_sample_data_path() + "/images/chessboard_01.jpg"


## Directory Structure
```
xrobot_calibration/
├── xrobot_calibration/          # Main package directory
│   ├── camera_calib/            # Submodule for camera calibration
│   │   ├── __init__.py
│   │   ├── calibration.py       # High-level interfaces
│   │   ├── camera_calibrator.py # CameraCalibrator class
│   │   ├── intrinsic.py         # Intrinsic calibration logic and load/save functions
│   │   ├── extrinsic.py         # Extrinsic calibration logic and load/save functions
│   │   ├── file_io.py           # Shared file I/O for saving/loading
│   │   ├── utils.py             # Camera-specific utilities
│   │   ├── pattern_detection.py  # Specific file for pattern detectio
│   │   └── ...
│   ├── arm_calib/               # Submodule for robotic arm calibration
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── utils.py             # Arm-specific utilities
│   │   └── ...
│   ├── mobile_base_calib/       # Submodule for mobile base calibration
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── utils.py             # Mobile-base-specific utilities
│   │   └── ...
│   ├── __init__.py              # Top-level init for the package
│   ├── common/                  # Shared utility module
│   │   ├── __init__.py
│   │   ├── math_utils.py        # Math-related utilities
│   │   ├── io_utils.py          # I/O-related utilities
│   │   ├── transform_utils.py   # Transformation-related utilities
│   │   └── ...
│   └── ...
├── notebooks/                   # Jupyter notebooks for tutorials and visualization
│   ├── camera_calibration_demo.ipynb
│   ├── arm_calibration_demo.ipynb
│   ├── mobile_base_calibration_demo.ipynb
│   └── visualization_tools.ipynb
├── data/                       # Sample data folder
│   ├── images/                 # Sample images for camera calibration
│   │   ├── chessboard_01.jpg
│   │   ├── chessboard_02.jpg
│   │   └── ...
│   ├── point_clouds/           # Example point clouds
│   │   ├── object_01.pcd
│   │   ├── object_02.pcd
│   │   └── ...
│   ├── configs/                # Example configuration files
│   │   ├── camera_config.json
│   │   ├── arm_config.yaml
│   │   └── ...
│   └── mobile_base/            # Mobile base-specific sample data
│       ├── odometry_log.csv
│       ├── wheel_config.json
│       └── ...
├── tests/                       # Test cases for all submodules
│   ├── camera_calib/
│   │   └── test_calibration.py
│   ├── arm_calib/
│   │   └── test_calibration.py
│   ├── mobile_base_calib/
│   │   └── test_calibration.py
│   └── ...
├── docs/                        # Documentation
│   └── index.md
├── examples/                    # Examples for usage
│   └── example_usage.py
├── setup.py                     # Setup script for PyPI
├── pyproject.toml               # Build system configuration
├── LICENSE                      # License file
├── README.md                    # Project description
├── .gitignore                   # Files to ignore in Git
└── requirements.txt             # Dependencies
```