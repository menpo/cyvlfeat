# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.


import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.string cimport memcpy

# Import the header files
from cyvlfeat._vl.quickshift cimport *
from cyvlfeat._vl.host cimport *


@cython.boundscheck(False)
cpdef cy_quickshift(np.ndarray[double, ndim=2, mode='c'] image,
                    int kernel_size, int max_dist, bint compute_estimate,
                    bint medoid, bint verbose):
    cdef:
        int *parentsi
        double sigma = kernel_size
        double tau = max_dist
        int ndims = image.ndim
        VlQS *q

        int n1 = image.shape[0]
        int n2 = image.shape[1]

        # int num_channels = image.shape[2] if ndims == 3 else 1
        int num_channels = 1

        # parent_c (same size as I) - Intermediate array
        np.ndarray[double, ndim=2, mode='c'] parents_c = np.empty(
            (n1*n2, 1), dtype=np.float64, order='C')

        # Create output arrays.
        # parents (same size as I)
        np.ndarray[double, ndim=2, mode='c'] parents = np.empty(
            (n1, n2), dtype=np.float64, order='C')

        # dists (same size as I)
        np.ndarray[double, ndim=2, mode='c'] dists = np.empty(
            (n1, n2), dtype=np.float64, order='C')

        # density (same size as I)
        np.ndarray[double, ndim=2, mode='c'] density = np.empty(
            (n1, n2), dtype=np.float64, order='C')


    if verbose:
        printf("quickshift:   [N1,N2,K]:             = [%d, %d, %d] \n", n1, n2, num_channels)
        printf("quickshift:   type:                  = %s\n", 'medoid' if medoid else 'quick')
        printf("quickshift:   kernel size:           = %g\n", sigma)
        printf("quickshift:   maximum gap:           = %g\n", tau)

    q = vl_quickshift_new(&image[0,0], n1, n2, num_channels)
    vl_quickshift_set_kernel_size(q, sigma)
    vl_quickshift_set_max_dist(q, tau)
    vl_quickshift_set_medoid(q, medoid)

    vl_quickshift_process(q)

    parentsi = vl_quickshift_get_parents(q)

    # Copy results
    for i in range(n1*n2):
        parents_c[i][0] = parentsi[i] + 1

    parents = np.reshape(parents_c, (n1, n2))

    memcpy(&dists[0, 0], vl_quickshift_get_dists(q), sizeof(double) * n1 * n2)

    if compute_estimate:
        memcpy(&density[0, 0], vl_quickshift_get_density(q), sizeof(double) * n1 * n2)

    vl_quickshift_delete(q)

    if compute_estimate:
        return parents, dists, density
    else:
        return parents, dists