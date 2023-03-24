
from glob import glob
import os
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


####### Metadata ########
NAME = "plastic-balanced-network"
DESCRIPTION = "Plastic Balanced Network Package (Akil et al., 2021)"
LDESCRIPTION = "This is a package with functions to simulate a plastic balanced network as defined in Akil et al., 2021."
# URL = text/markdown
MAINTAINER = "Alan Akil"
EMAIL = "alan.akil@yahoo.com"
REQUIRES_PYTHON = ">=3.7,<3.10"
PACKAGES = find_packages(where = "src", exclude=("tests",))

ROOT_DIR = Path(__file__).parent

PACKAGE_DIR = os.path.join(ROOT_DIR, "src", NAME)
about = {}
with open(os.path.join(PACKAGE_DIR, "VERSION")) as f:
    _version = f.read().strip()
    about["__version__"] = _version

def list_reqs(fname="requirements.txt"):
    with open(os.path.join(ROOT_DIR, fname)) as req:
        return req.read().splitlines()

EXTRAS_REQUIRE = {
    "tests": ["coverage", "pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-rtd-theme"]
}

EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["docs"]


setup(
    name = NAME,
    version = about["__version__"],
    description = DESCRIPTION,
    long_description = LDESCRIPTION,
    long_description_content_type = "text/markdown",
    maintainer=MAINTAINER,
    maintainer_email=EMAIL,
    # url=URL,
    packages=PACKAGES,
    package_dir={"":"src"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    classifiers = [
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Python",
        "Operating System :: Python :: 3",
        "Operating System :: Python :: 3.8",
        "Operating System :: Python :: 3.9",
        "Operating System :: Python :: 3.10",
        "Operating System :: Python :: Implementation :: CPython",
        "Operating System :: Python :: Implementation :: PyPy",
    ],
    platforms=["Windows", "Linux", "Mac OS-X", "UNIX"],
    python_requires = REQUIRES_PYTHON,
    install_requires = list_reqs(),
    extras_require = EXTRAS_REQUIRE,
    setup_requires = [
        "pytest-runner"
    ],
)
