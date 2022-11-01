import os
from pathlib import Path
from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.7 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the
# main shipdetect __init__ can detect if it is being loaded by the setup
# routine, to avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by ship-detect to
# recursively build the compiled extensions in sub-packages is based on the
# Python import machinery.
builtins.__SHIPDETECT_SETUP__ = True


def get_long_description() -> str:
    CURRENT_DIR = Path(__file__).parent
    return (CURRENT_DIR / "README.rst").read_text(encoding="utf8")


import shipdetect._min_dependencies as min_deps

ver_file = os.path.join("shipdetect", "_version.py")
with open(ver_file) as f:
    exec(f.read())

MAINTAINER = "J. Fonseca"
MAINTAINER_EMAIL = "jpfonseca@novaims.unl.pt"
URL = "about:blank"
VERSION = __version__
SHORT_DESCRIPTION = (
    "Implementation of Machine Learning algorithms, experiments and utilities."
)
LICENSE = "MIT"
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
INSTALL_REQUIRES = (min_deps.tag_to_packages["install"],)
EXTRAS_REQUIRE = {
    key: value
    for key, value in min_deps.tag_to_packages.items()
    if key != "install"
}

setup(
    name="ship-detect",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=URL,
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
