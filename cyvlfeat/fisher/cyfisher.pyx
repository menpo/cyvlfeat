# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython

# Import the header files
from cyvlfeat._vl.fisher cimport vl_fisher_encode
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_NORMALIZED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_SQUARE_ROOT
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_IMPROVED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_FAST

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
        vl_size numClusters = MEANS.shape[1]
        vl_size dimension = MEANS.shape[0]
        vl_size numData = X.shape[1]
        int flags = 0

    if Normalized:
        flags |= VL_FISHER_FLAG_NORMALIZED
    
    if SquareRoot:
        flags |= VL_FISHER_FLAG_SQUARE_ROOT
        
    if Improved:
        flags |= VL_FISHER_FLAG_IMPROVED
        
    if Fast:
        flags |= VL_FISHER_FLAG_FAST
    
    if Verbose:
        print('vl_fisher: num data: %d' % numData)
        print('vl_fisher: num clusters: %d' % numClusters)
        print('vl_fisher: data dimension: %d' % dimension)
        print('vl_fisher: code dimension: %d' % 2 * numClusters * dimension)
        print('vl_fisher: square root: %d' % SquareRoot)
        print('vl_fisher: normalized: %d' % Normalized)
        print('vl_fisher: fast: %d' % Fast)

    cdef float[:] enc = np.zeros((2*numClusters*dimension,),dtype=np.float32)
    cdef int numTerms = vl_fisher_encode(&enc.types,
                                         dataType,
                                         MEANS.ctype,
                                         dimension,
                                         numClusters,
                                         COVARIANCES.ctypes,
                                         PRIORS.ctypes,
                                         X.ctypes,
                                         numData,
                                         flags)

    if Verbose:
        print('vl_fisher: sparsity of assignments: %.2f%% (%d non-negligible assignments)' \
              % (100.0 * (1.0 - np.float32(numTerms)/(numData*numClusters+1e-12)),numTerms))

    return enc