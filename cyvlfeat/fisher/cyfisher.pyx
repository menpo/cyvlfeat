# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
from libc.stdio cimport printf
from libc.stdlib cimport qsort

# Import the header files
from cyvlfeat._vl.fisher cimport *
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.mathop cimport VL_PI

@cython.boundscheck(False)
cpdef cy_fisher(np.ndarray[float, ndim=2, mode='c'] X,
                np.ndarray[float, ndim=2, mode='c'] MEANS,
                np.ndarray[float, ndim=2, mode='c'] COVARIANCES,
                np.ndarray[float, ndim=1, mode='c'] PRIORS,
                bint Normalized,
                bint SquareRoot,
                bint Improved,
                bint Fast,
                bint Verbose):

    cdef:
        int flags = 0;

    vl_fisher_encode()
#(void * enc, vl_type dataType,
 #void const * means, vl_size dimension, vl_size numClusters,
 #void const * covariances,
 #void const * priors,
 #void const * data, vl_size numData,
 #int flags) ;

    ENC = None
    return ENC