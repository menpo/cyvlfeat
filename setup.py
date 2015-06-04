from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os.path as op
import os
import platform
import fnmatch
import shutil


def walk_for_package_data(ext_pattern):
    paths = []
    for root, dirnames, filenames in os.walk('cyvlfeat'):
        for filename in fnmatch.filter(filenames, ext_pattern):
            # Slice cyvlfeat off the beginning of the path
            paths.append(
                op.relpath(os.path.join(root, filename), 'cyvlfeat'))
    return paths


IS_WIN = platform.system() == 'Windows'
IS_CONDA = os.environ.get('CONDA_BUILD', False)

include_dirs = []
library_dirs = []

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
    
    include_dirs.append(os.environ['LIBRARY_INC'])
    library_dirs.append(conda_bin_dir)
    
    # On Windows, there is no relative linking against DLLS,
    # so every extension MUST have a copy of the vl.dll next to it.
    sift_dll_path = op.join('cyvlfeat', 'sift', 'vl.dll')
    shutil.copy(conda_vl_dll_path, sift_dll_path)
    
vl_extensions = [
    Extension('cyvlfeat.sift.cysift',
              [op.join('cyvlfeat', 'sift', 'cysift.pyx')],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=['vl'], 
              language='c++')
]

# Grab all the pyx and pxd Cython files for uploading to pypi
cython_files = walk_for_package_data('*.p[xy][xd]')

# Move the dlls next to the package if on Windows
if IS_WIN:
    dll_paths = walk_for_package_data('*.dll')
else:
    dll_paths = []
            
setup(
    name='cyvlfeat',
    version='0.3',
    description='Cython wrapper of the VLFeat toolkit',
    url='https://github.com/menpo/cyvlfeat/',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(vl_extensions),
    packages=find_packages(),
    package_data={'cyvlfeat': cython_files + dll_paths}
)

