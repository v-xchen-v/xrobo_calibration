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
│   │   ├── calibration.py
│   │   ├── utils.py
│   │   └── ...
│   ├── arm_calib/               # Submodule for robotic arm calibration
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── utils.py
│   │   └── ...
│   ├── mobile_base_calib/       # Submodule for mobile base calibration
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── utils.py
│   │   └── ...
│   ├── __init__.py              # Top-level init for the package
│   └── common/                  # Shared utilities
│       ├── __init__.py
│       ├── math_utils.py
│       ├── io_utils.py
│       └── ...
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