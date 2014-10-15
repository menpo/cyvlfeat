from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os.path as op
import os
import fnmatch
from glob import glob
import platform


vl_feat_sources = [s for s in glob(op.join('cyvlfeat', 'vlfeat', 'vl', '*.c'))]

if platform.system() == 'Windows':
    compile_args = ['/DVL_BUILD_DLL', '/DDISABLE_OPENMP', 
			        '/D__SSE2__',
	                '/MD', '/D_CRT_SECURE_NO_DEPRECATE', 
					'/D__LITTLE_ENDIAN__', '/DVL_DISABLE_AVX']
else:  # Assume unix
    compile_args = ['-DDISABLE_OPENMP=1', '-mavx']


extensions = [
    Extension('cysift', [op.join('cyvlfeat', 'sift', 'cysift.pyx')] + vl_feat_sources,
              include_dirs=['vlfeat'],
              extra_compile_args=compile_args
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
    version='0.2',
    description='Cython wrapper of the VLFeat toolkit',
    url='https://github.com/menpo/cyvlfeat/',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    package_data={'cyvlfeat': ['vlfeat/vl/*.th', 'vlfeat/vl/*.tc',
                               'vlfeat/vl/*.h', 'vlfeat/vl/*.c'] + cython_files}
)

