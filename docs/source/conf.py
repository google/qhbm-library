# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../"))

import myst_parser
import qhbmlib
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'QHBM Library'
copyright = '2022, The QHBM Library Authors'
author = 'The QHBM Library Authors'

# The full version, including alpha/beta/rc tags
release = '0.3.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'myst_parser',
  'sphinx.ext.autodoc',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Add .rst for the index and .md for other documents
source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
