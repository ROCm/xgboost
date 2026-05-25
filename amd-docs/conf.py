# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path

RELEASE_VERSION = "3.2.0"
DOCS_DIR = Path(__file__).parent.resolve()
# ROOT_DIR = DOCS_DIR.parent

project = "XGBoost"
project_path = str(DOCS_DIR).replace("\\", "/")
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
version = RELEASE_VERSION
release = RELEASE_VERSION
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

extensions = [
    "rocm_docs",
]
external_toc_path = "./sphinx/_toc.yml"
external_projects_remote_repository = "xgboost"

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm-finance",
}
html_title = f"XGBoost {RELEASE_VERSION} documentation"
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}
