from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os.path as op
import os
import platform
import fnmatch
import versioneer


INCLUDE_DIRS = []
LIBRARY_DIRS = []


def walk_for_package_data(ext_pattern):
    paths = []
    for root, dirnames, filenames in os.walk('cyvlfeat'):
        for filename in fnmatch.filter(filenames, ext_pattern):
            # Slice cyvlfeat off the beginning of the path
            paths.append(
                op.relpath(os.path.join(root, filename), 'cyvlfeat'))
    return paths


def gen_extension(path_name, sources):
    global INCLUDE_DIRS, LIBRARY_DIRS
    return Extension(
        path_name,
        sources=sources,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=['vl'],
        language='c'
    )


IS_WIN = platform.system() == 'Windows'
IS_CONDA = os.environ.get('CONDA_BUILD', False)

# If we are building from the conda folder,
# then we know we can manually copy some files around
# because we have control of the setup. If you are
# building this manually or pip installing, you must satisfy
# that the vlfeat vl folder is on the PATH (for the headers)
# and that the vl.dll file is visible to the build system
# as well.
if IS_WIN and IS_CONDA:
    conda_bin_dir = os.environ['LIBRARY_BIN']
    conda_vl_dll_path = op.join(conda_bin_dir, 'vl.dll')
    INCLUDE_DIRS.append(os.environ['LIBRARY_INC'])
    LIBRARY_DIRS.append(conda_bin_dir)

vl_extensions = [
    gen_extension('cyvlfeat.misc.cylbp',
                  [op.join('cyvlfeat', 'misc', 'cylbp.pyx')]),
    gen_extension('cyvlfeat.sift.cysift',
                  [op.join('cyvlfeat', 'sift', 'cysift.pyx')]),
    gen_extension('cyvlfeat.fisher.cyfisher',
                  [op.join('cyvlfeat', 'fisher', 'cyfisher.pyx')]),
    gen_extension('cyvlfeat.hog.cyhog',
                  [op.join('cyvlfeat', 'hog', 'cyhog.pyx')]),
    gen_extension('cyvlfeat.kmeans.cykmeans',
                  [op.join('cyvlfeat', 'kmeans', 'cykmeans.pyx')]),
    gen_extension('cyvlfeat.generic.generic',
                  [op.join('cyvlfeat', 'generic', 'generic.pyx')])
]

# Grab all the pyx and pxd Cython files for uploading to pypi
cython_files = walk_for_package_data('*.p[xy][xd]')
            
setup(
    name='cyvlfeat',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Cython wrapper of the VLFeat toolkit',
    url='https://github.com/menpo/cyvlfeat',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(vl_extensions),
    packages=find_packages(),
    package_data={'cyvlfeat': cython_files}
)
