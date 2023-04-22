import os
from pathlib import Path
import sys

NAME = 'plastic_balanced_network'
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = Path(__file__).parent
PACKAGE_DIR = os.path.join(ROOT_DIR, "src", NAME)
about = {}
with open(os.path.join(PACKAGE_DIR, "VERSION")) as f:
    _version = f.read().strip()
    about["__version__"] = _version

sys.path.append(ROOT_DIR)
sys.path.append(PACKAGE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

import plastic_balanced_network

# -- Project information -----------------------------------------------------

project = "Plastic Balanced Networks"
author = "Alan Akil"
version = release = about["__version__"]

# master_doc = 'index'

# -- General configuration ---------------------------------------------------

html_theme = 'sphinx_rtd_theme'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_use_smartypants = True
html_last_updated_fmt = "%Y, %b, %d"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"]
}
html_short_title = "%s-%s" % (project, version)
