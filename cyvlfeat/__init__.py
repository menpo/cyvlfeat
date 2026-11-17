import cyvlfeat.fisher
import cyvlfeat.generic
import cyvlfeat.gmm
import cyvlfeat.hog
import cyvlfeat.kmeans
import cyvlfeat.quickshift
import cyvlfeat.sift


try:
    from ._version import __version__
except ImportError:  # running from a source tree that has not been built or installed
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("cyvlfeat")
    except PackageNotFoundError:
        __version__ = "unknown"
