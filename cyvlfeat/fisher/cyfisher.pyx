# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython

# Import the header files
from cyvlfeat.cy_util cimport dtype_from_memoryview
from cyvlfeat._vl.host cimport VL_TYPE_FLOAT, VL_TYPE_DOUBLE
from cyvlfeat._vl.host cimport vl_size
from cyvlfeat._vl.fisher cimport vl_fisher_encode
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_NORMALIZED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_SQUARE_ROOT
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_IMPROVED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_FAST


@cython.boundscheck(False)
cpdef cy_fisher(cython.floating[:, ::1] X,
                cython.floating[:, ::1] means,
                cython.floating[:, ::1] covariances,
                cython.floating[::1] priors,
                bint normalized,
                bint square_root,
                bint improved,
                bint fast,
                bint verbose):
    dtype = dtype_from_memoryview(X)

    cdef:
        vl_size n_clusters = means.shape[0]
        vl_size n_dimensions = means.shape[1]
        vl_size n_data = X.shape[0]
        int flags = 0
        int num_terms = 0
        int vl_float_type = 0
        cython.floating[::1] enc = np.zeros(2 * n_clusters * n_dimensions,
                                            dtype=dtype)

    if normalized:
        flags |= VL_FISHER_FLAG_NORMALIZED

    if square_root:
        flags |= VL_FISHER_FLAG_SQUARE_ROOT

    if improved:
        flags |= VL_FISHER_FLAG_IMPROVED

    if fast:
        flags |= VL_FISHER_FLAG_FAST

    if verbose:
        print('vl_fisher: num data:       {}\n'
              'vl_fisher: num clusters:   {}\n'
              'vl_fisher: data dimension: {}\n'
              'vl_fisher: code dimension: {}\n'
              'vl_fisher: square root:    {}\n'
              'vl_fisher: normalized:     {}\n'
              'vl_fisher: fast:           {}'.format(
            n_data, n_clusters, n_dimensions, 2 * n_clusters * n_dimensions,
            square_root, normalized, fast))

    vl_float_type = VL_TYPE_FLOAT if dtype == np.float32 else VL_TYPE_DOUBLE
    num_terms = vl_fisher_encode(&enc[0],
                                 vl_float_type,
                                 &means[0, 0],
                                 n_dimensions,
                                 n_clusters,
                                 &covariances[0, 0],
                                 &priors[0],
                                 &X[0, 0],
                                 n_data,
                                 flags)

    if verbose:
        print('vl_fisher: sparsity of assignments: {:.2f}% '
              '({} non-negligible assignments)'.format(
            100.0 * (1.0 - <float>num_terms / (n_data * n_clusters + 1e-12)),
            num_terms))

    return np.asarray(enc)
