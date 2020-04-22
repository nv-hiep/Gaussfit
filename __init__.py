import sys
from distutils.version import LooseVersion

__minimum_python_version__ = "3.6"

__all__ = []


class UnsupportedPythonError(Exception):
    pass


if LooseVersion(sys.version) < LooseVersion(__minimum_python_version__):
    raise UnsupportedPythonError("gausspy does not support Python < {}"
                                 .format(__minimum_python_version__))