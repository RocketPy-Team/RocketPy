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

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "RocketPy"
copyright = "2020, RocketPy Team"

author = "Giovani Hidalgo Ceotto"

# The full version, including alpha/beta/rc tags
release = "0.11.0"


# -- General configuration ---------------------------------------------------
master_doc = "index"
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "m2r2",
]

# source_suffix = '.rst'
# source_suffix = ['.rst', '.md']

# Don't run notebooks
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

napoleon_numpy_docstring = True
autodoc_member_order = "bysource"
autoclass_content = "both"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]
html_css_files = ["notebooks.css"]
html_logo = "static/RocketPy_Logo_Black.svg"
html_favicon = "static/favicon.ico"
html_theme_options = {
    "logo_link": "index",
    "github_url": "https://github.com/Projeto-Jupiter/RocketPy",
    "collapse_navigation": True,
    "show_toc_level": 3,
}

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_theme_options = {"navbar_end": ["navbar-icon-links.html", "search-field.html"]}

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "rocketpy"
