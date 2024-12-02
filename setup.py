from setuptools import setup, find_packages

setup(
    name='xrobo_calibration',
    version='0.1.0',
    author='Xi Chen',
    author_email='xichen0907@gmail.com',
    description='The xrobo_calibration project provides a robust framework for calibrating robotic systems. It supports tools and methodologies for determining transformations between key components, such as cameras, robotic arms, and mobile base using calibration patterns like chessboards or markers. ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/v-xchen-v/xrobo_calibration',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'opencv-python',
        'imutils',
    ],
    extras_require={
        # Optional dependencies for development
        "dev":[
            "build", # For building the package
            "twine", # For uploading the package,
            "pytest", # For running tests,
            "sphinx", # For generating documentation,
            "sphinx_rtd_theme", # For a nice theme in the documentation
            "sphinx-autodoc-typehints", # For type hints in the documentation
            "myst-parser", # For markdown support in the documentation
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.10',
)
    