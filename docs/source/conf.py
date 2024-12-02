# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust to your project’s root directo


# -- Project information -----------------------------------------------------
project = 'xrobo_calibration'
copyright = '2024, Xi Chen'
author = 'Xi Chen'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Auto-generate API docs
    'sphinx.ext.napoleon',       # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',       # Link to source code
    'sphinx_autodoc_typehints',  # Add type hints to docs
    'myst_parser',              # Support for Markdown files
]

templates_path = ['_templates']
exclude_patterns = []


source_suffix = {
   '.rst': 'restructuredtext',
   '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
