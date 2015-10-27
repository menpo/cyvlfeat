# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

cdef extern from "vl/fisher.h":
    ctypedef unsigned int vl_type
    ctypedef unsigned int vl_size
    vl_size vl_fisher_encode(void* enc,
                             vl_type dataType,
                             void* means,
                             vl_size dimension,
                             vl_size numClusters,
                             void* covariances,
                             void* priors,
                             void* data,
                             vl_size numData,
                             int flags)
    cdef int VL_TYPE_FLOAT "VL_TYPE_FLOAT"
    cdef int VL_FISHER_FLAG_SQUARE_ROOT "VL_FISHER_FLAG_SQUARE_ROOT"
    cdef int VL_FISHER_FLAG_NORMALIZED "VL_FISHER_FLAG_NORMALIZED"
    cdef int VL_FISHER_FLAG_IMPROVED "VL_FISHER_FLAG_IMPROVED"
    cdef int VL_FISHER_FLAG_FAST "VL_FISHER_FLAG_FAST"
