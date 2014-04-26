from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os.path as op
from glob import glob


vl_feat_sources = []
for s in glob(op.join('vlfeat', 'vl', '*.c')):
    vl_feat_sources.append(s)

extensions = [
    Extension("cysift", ["cyvlfeat/sift/cysift.pyx"] + vl_feat_sources,
              include_dirs=['vlfeat'],
              extra_compile_args=['-DDISABLE_OPENMP=1', '-DVL_DISABLE_AVX=1']
    )
]

setup(
    name='cyvlfeat',
    version='0.2',
    description='Cython wrapper of the VLFeat toolkit',
    author='Patrick Snape',
    author_email='p.snape@imperial.ac.uk',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions, language='c++'),
    packages=find_packages()
)
