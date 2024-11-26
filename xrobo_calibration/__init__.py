# xrobo_calibration/__init__.py
# This file is the entry point for the xrobo_calibration package. It is used to exposure the package's functionality to the outside world. In this file, we define the package's version number and import the functions from the submodules. This allows users to access the functions directly from the package namespace.
__version__ = "0.1.0"

from .camera_calib import calibrate_camera
from .arm_calib import calibrate_arm
from .mobile_base_calib import calibrate_mobile_base