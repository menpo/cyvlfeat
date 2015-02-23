from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os.path as op
import os
import fnmatch

extensions = [
    Extension('cyvlfeat.sift.cysift',
              [op.join('cyvlfeat', 'sift', 'cysift.pyx')],
              libraries=['vl']
    )
]

# Grab all the pyx and pxd Cython files for uploading to pypi
cython_files = []
for root, dirnames, filenames in os.walk('cyvlfeat'):
    for filename in fnmatch.filter(filenames, '*.p[xy][xd]'):
        # Slice cyvlfeat off the beginning of the path
        cython_files.append(
            op.relpath(os.path.join(root, filename), 'cyvlfeat'))

setup(
    name='cyvlfeat',
    version='0.3',
    description='Cython wrapper of the VLFeat toolkit',
    url='https://github.com/menpo/cyvlfeat/',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    package_data={'cyvlfeat': cython_files}
)

