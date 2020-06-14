import numpy as np
cimport numpy as np
cimport cython
# import cython
from libc.stdio cimport printf

# Import the header files
from cyvlfeat.cy_util cimport dtype_from_memoryview
from cyvlfeat._vl.host cimport VL_TYPE_FLOAT, VL_TYPE_DOUBLE
from cyvlfeat._vl.host cimport vl_size
from cyvlfeat._vl.vlad cimport vl_vlad_encode
from cyvlfeat._vl.vlad cimport VL_VLAD_FLAG_NORMALIZE_COMPONENTS
from cyvlfeat._vl.vlad cimport VL_VLAD_FLAG_SQUARE_ROOT
from cyvlfeat._vl.vlad cimport VL_VLAD_FLAG_UNNORMALIZED
from cyvlfeat._vl.vlad cimport VL_VLAD_FLAG_NORMALIZE_MASS

@cython.boundscheck(False)
cpdef cy_vlad(cython.floating[:, ::1] X,
            cython.floating[:, ::1] means,
            cython.floating[:, ::1] assignments,
            bint unnormalized,
            bint square_root,
            bint normalize_components,
            bint normalize_mass,
            bint verbose):
    dtype = dtype_from_memoryview(X)
    cdef:
        vl_size n_clusters = means.shape[0]
        vl_size n_dimensions = means.shape[1]
        vl_size n_data = X.shape[0]
        int flags = 0
        cython.floating[::1] enc = np.zeros(n_clusters * n_dimensions, dtype=dtype)

    if unnormalized:
        flags |= VL_VLAD_FLAG_UNNORMALIZED

    if normalize_components:
        flags |= VL_VLAD_FLAG_NORMALIZE_COMPONENTS

    if normalize_mass:
        flags |= VL_VLAD_FLAG_NORMALIZE_MASS

    if square_root:
        flags |= VL_VLAD_FLAG_SQUARE_ROOT

    if verbose:
        # check for 2 * n_clusters * n_dimensions in print spree
        printf("vl_vlad: num data:         %d\n",
                   n_data)
        printf("vl_vlad: num clusters:           %d\n",
                   n_clusters)
        printf("vl_vlad: data dimension:           %d\n",
                   n_dimensions)
        printf("vl_vlad: code dimension:           %d\n",
                   2 * n_clusters * n_dimensions)
        printf("vl_vlad: unnormalized:         %d\n",
                   unnormalized)
        printf("vl_vlad: normalize mass:           %d\n",
                   normalize_mass)
        printf("vl_vlad: normalize components:           %d\n",
                   normalize_components)
        printf("vl_vlad: square root:           %d\n",
                   square_root)

    vl_float_type = VL_TYPE_FLOAT if dtype == np.float32 else VL_TYPE_DOUBLE
    vl_vlad_encode(&enc[0],
                  vl_float_type,
                  &means[0, 0],
                  n_dimensions,
                  n_clusters,
                  &X[0, 0],
                  n_data,
                  &assignments[0, 0],
                  flags)

    return np.asarray(enc)
