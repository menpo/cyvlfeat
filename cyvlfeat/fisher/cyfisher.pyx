# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
import ctypes
cimport numpy as np
cimport cython

# Import the header files
from cyvlfeat._vl.host cimport VL_TYPE_FLOAT, VL_TYPE_DOUBLE
from cyvlfeat._vl.host cimport vl_size, vl_type
from cyvlfeat._vl.fisher cimport vl_fisher_encode
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_NORMALIZED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_SQUARE_ROOT
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_IMPROVED
from cyvlfeat._vl.fisher cimport VL_FISHER_FLAG_FAST


@cython.boundscheck(False)
cpdef cy_fisher(np.ndarray X,
                np.ndarray MEANS,
                np.ndarray COVARIANCES,
                np.ndarray PRIORS,
                bint normalized,
                bint square_root,
                bint improved,
                bint fast,
                bint verbose):

    cdef:
        vl_size n_clusters = MEANS.shape[0]
        vl_size n_dimensions = MEANS.shape[1]
        vl_size n_data = X.shape[0]
        void* data
        void* means_data
        void* covariances_data
        void* priors_data
        void* envc_data
        np.ndarray enc

        int flags = 0

        vl_type dataType
    if X.dtype == "float32":
        dataType = VL_TYPE_FLOAT
    elif X.dtype == "float64":
        dataType = VL_TYPE_DOUBLE
    else:
        raise TypeError("Unsupported data type for X")

    if MEANS.dtype != X.dtype:
        raise TypeError("MEANS does not have the same type as X")

    if COVARIANCES.dtype != X.dtype:
        raise TypeError("COVARIANCES does not have the same type as X")

    if PRIORS.dtype != X.dtype:
        raise TypeError("PRIORS does not have the same type as X")

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

    enc = np.zeros(2 * n_clusters * n_dimensions,
                                 dtype=X.dtype)


    data = <void*> X.data
    means_data = <void*> MEANS.data
    covariances_data = <void*> COVARIANCES.data
    priors_data = <void*> PRIORS.data
    enc_data = <void*> enc.data

    cdef int num_terms = vl_fisher_encode(enc_data,
                                          dataType,
                                          means_data,
                                          n_dimensions,
                                          n_clusters,
                                          covariances_data,
                                          priors_data,
                                          data,
                                          n_data,
                                          flags)

    if verbose:
        print('vl_fisher: sparsity of assignments: {:.2f}% '
              '({} non-negligible assignments)'.format(
            100.0 * (1.0 - <float>num_terms / (n_data * n_clusters + 1e-12)),
            num_terms))

    return enc
