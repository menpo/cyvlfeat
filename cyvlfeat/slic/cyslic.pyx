# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).


import numpy as np
cimport numpy as np
import ctypes
cimport cython
import cython
from libc.stdio cimport printf

# Import the header files
from cyvlfeat._vl.slic cimport *
from cyvlfeat._vl.host cimport *

@cython.boundscheck(False)
cpdef cy_slic(np.ndarray[float, ndim=2, mode='c'] image, int region_size,
              float regularizer, bint verbose):

    cdef:
        vl_size width = image.shape[1]
        vl_size height = image.shape[0]
        vl_size n_channels = 1
        # vl_size n_channels = image.shape[2]
        int min_region_size = -1
        np.ndarray[unsigned int, ndim=2, mode='c'] segmentation

    if min_region_size < 0:
        min_region_size = region_size * region_size/36
        # printf("min_reigon_size cannot be less than 0. Assigning min_reigon_size = %d\n", min_region_size)

    if verbose:
        print('vl_slic: image:                  [{} x {} X {}]\n'
              'vl_slic: region size:            {}\n'
              'vl_slic: regularizer:            {}\n'
              'vl_slic: min region size:        {}'. format(
            width, height, n_channels, region_size, regularizer, min_region_size ))

    segmentation = np.zeros((height, width), dtype=np.uint32, order='C')

    vl_slic_segment(&segmentation[0,0], &image[0, 0], height, width, n_channels,
                    region_size, regularizer, min_region_size)

    return np.asarray(segmentation)
