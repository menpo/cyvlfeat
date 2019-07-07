import os
import platform
import site

from setuptools import setup, find_packages, Extension

import versioneer

SYS_PLATFORM = platform.system().lower()
IS_LINUX = 'linux' in SYS_PLATFORM
IS_OSX = 'darwin' == SYS_PLATFORM
IS_WIN = 'windows' == SYS_PLATFORM


# Get Numpy include path without importing it
NUMPY_INC_PATHS = [os.path.join(r, 'numpy', 'core', 'include')
                   for r in site.getsitepackages() if
                   os.path.isdir(os.path.join(r, 'numpy', 'core', 'include'))]
if len(NUMPY_INC_PATHS) == 0:
    try:
        import numpy as np
    except ImportError:
        raise ValueError("Could not find numpy include dir and numpy not installed before build - "
                         "cannot proceed with compilation of cython modules.")
    else:
        # just ask numpy for it's include dir
        NUMPY_INC_PATHS = [np.get_include()]

elif len(NUMPY_INC_PATHS) > 1:
    print("Found {} numpy include dirs: "
          "{}".format(len(NUMPY_INC_PATHS), ', '.join(NUMPY_INC_PATHS)))
    print("Taking first (highest precedence on path): {}".format(
        NUMPY_INC_PATHS[0]))
NUMPY_INC_PATH = NUMPY_INC_PATHS[0]


# ---- C/C++ EXTENSIONS ---- #
# Stolen (and modified) from the Cython documentation:
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html
def no_cythonize(extensions, **_ignore):
    import os.path as op
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
                if not op.exists(sfile):
                    raise ValueError('Cannot find pre-compiled source file '
                                     '({}) - please install Cython'.format(sfile))
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


def build_extension_from_pyx(pyx_path, extra_sources_paths=None):
    # If we are building from the conda folder,
    # then we know we can manually copy some files around
    # because we have control of the setup. If you are
    # building this manually or pip installing, you must satisfy
    # that the vlfeat vl folder is on the PATH (for the headers)
    # and that the vl.dll file is visible to the build system
    # as well.
    include_dirs = [NUMPY_INC_PATH]
    library_dirs = []
    if IS_WIN:
        include_dirs.append(os.environ['LIBRARY_INC'])
        library_dirs.append(os.environ['LIBRARY_BIN'])

    if extra_sources_paths is None:
        extra_sources_paths = []
    extra_sources_paths.insert(0, pyx_path)
    ext = Extension(name=pyx_path[:-4].replace('/', '.'),
                    sources=extra_sources_paths,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    libraries=['vl'],
                    language='c')
    if IS_LINUX or IS_OSX:
        ext.extra_compile_args.append('-Wno-unused-function')
    if IS_OSX:
        ext.extra_link_args.append('-headerpad_max_install_names')
    return ext


try:
    from Cython.Build import cythonize
except ImportError:
    import warnings
    cythonize = no_cythonize
    warnings.warn('Unable to import Cython - attempting to build using the '
                  'pre-compiled C++ files.')


cython_modules = [
    build_extension_from_pyx('cyvlfeat/generic/generic.pyx'),
    build_extension_from_pyx('cyvlfeat/fisher/cyfisher.pyx'),
    build_extension_from_pyx('cyvlfeat/gmm/cygmm.pyx'),
    build_extension_from_pyx('cyvlfeat/hog/cyhog.pyx'),
    build_extension_from_pyx('cyvlfeat/kmeans/cykmeans.pyx'),
    build_extension_from_pyx('cyvlfeat/sift/cysift.pyx')
]
cython_exts = cythonize(cython_modules, quiet=True)

setup(
    name='cyvlfeat',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Cython wrapper of the VLFeat toolkit',
    url='https://github.com/menpo/cyvlfeat',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    ext_modules=cython_exts,
    packages=find_packages()
)
