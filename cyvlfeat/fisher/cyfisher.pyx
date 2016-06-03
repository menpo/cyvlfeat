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

        np.ndarray[np.float32_t, ndim=1, mode="c"] encf
        np.ndarray[np.float32_t, ndim=2, mode="c"] meansf
        np.ndarray[np.float32_t, ndim=2, mode="c"] covarsf
        np.ndarray[np.float32_t, ndim=1, mode="c"] priorsf
        np.ndarray[np.float32_t, ndim=2, mode="c"] Xf

        np.ndarray[np.float64_t, ndim=1, mode="c"] encd
        np.ndarray[np.float64_t, ndim=2, mode="c"] meansd
        np.ndarray[np.float64_t, ndim=2, mode="c"] covarsd
        np.ndarray[np.float64_t, ndim=1, mode="c"] priorsd
        np.ndarray[np.float64_t, ndim=2, mode="c"] Xd

        int flags = 0
        int num_terms

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

    if X.dtype == "float32":

        encf = np.zeros(
            2 * n_clusters * n_dimensions, dtype=X.dtype
        )
        meansf = MEANS
        covarsf = COVARIANCES
        priorsf = PRIORS
        Xf = X

        num_terms = vl_fisher_encode(&encf[0],
                                     VL_TYPE_FLOAT,
                                     &meansf[0, 0],
                                     n_dimensions,
                                     n_clusters,
                                     &covarsf[0, 0],
                                     &priorsf[0],
                                     &Xf[0, 0],
                                     n_data,
                                     flags)
        enc = encf

    elif X.dtype == "float64":

        encd = np.zeros(
            2 * n_clusters * n_dimensions, dtype=X.dtype
        )
        meansd = MEANS
        covarsd = COVARIANCES
        priorsd = PRIORS
        Xd = X

        num_terms = vl_fisher_encode(&encd[0],
                                     VL_TYPE_DOUBLE,
                                     &meansd[0, 0],
                                     n_dimensions,
                                     n_clusters,
                                     &covarsd[0, 0],
                                     &priorsd[0],
                                     &Xd[0, 0],
                                     n_data,
                                     flags)
        enc = encd
    else:
        raise ValueError("The type of the input data X is not 'float32' nor "
                         "'float64'.")

    if verbose:
        print('vl_fisher: sparsity of assignments: {:.2f}% '
              '({} non-negligible assignments)'.format(
            100.0 * (1.0 - <float>num_terms / (n_data * n_clusters + 1e-12)),
            num_terms))

    return enc
