"""Sphinx configuration for VB-Mitigator."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "VB-Mitigator"
author = "Ioannis Sarridis"
copyright = "2026, Ioannis Sarridis"

try:
    from vbmitigator import __version__ as release
except Exception:  # pragma: no cover
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# MyST (Markdown) niceties: ::: admonitions, definition lists, auto anchors.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
autodoc_mock_imports = ["torch", "torchvision", "ram", "ollama", "transformers"]

# --- HTML output ---
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_title = "VB-Mitigator"
html_theme_options = {
    "logo_only": False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_nav_header_background": "#0e8f86",
}
templates_path = ["_templates"]
exclude_patterns = []
