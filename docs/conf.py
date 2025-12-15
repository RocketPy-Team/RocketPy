# pylint: skip-file

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

package_path = os.path.abspath("../")
sys.path.insert(0, package_path)
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# -- Project information -----------------------------------------------------

project = "RocketPy"
copyright = "2025, RocketPy Team"

author = "RocketPy Team"

# The full version, including alpha/beta/rc tags
release = "1.11.0"


# -- General configuration ---------------------------------------------------
master_doc = "index"
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinx_design",
    "jupyter_sphinx",
    "nbsphinx",
]


# Compatibility: https://about.readthedocs.com/blog/2024/07/addons-by-default/
## Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

## Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True

# Don't run notebooks
nbsphinx_execute = "never"

# Configure jupyter_sphinx execution behavior
jupyter_execute_kwargs = {
    "timeout": 300,  # 5 minutes timeout per cell
    "allow_errors": True,  # Continue building even if cells raise errors
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autodoc_member_order = "bysource"
autoclass_content = "class"

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract type hints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
numfig = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]
html_css_files = ["rocketpy.css"]
html_favicon = "static/favicon.ico"

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
}
html_theme_options = {
    "logo": {
        "image_light": "static/RocketPy_Logo_black.png",
        "image_dark": "static/RocketPy_Logo_white.png",
    },
    "navigation_with_keys": False,
    "collapse_navigation": True,
    "github_url": "https://github.com/RocketPy-Team/RocketPy",
    "navbar_end": ["theme-switcher", "navbar-icon-links.html"],
    "icon_links": [
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/rocketpy/",
            "icon": "fa-brands fa-linkedin",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/rocketpy/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "rocketpy"
