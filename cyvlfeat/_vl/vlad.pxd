# Copyright (C) 2013 David Novotny and Andera Vedaldi.
# All rights reserved.

# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

from cyvlfeat._vl.host cimport vl_size
from cyvlfeat._vl.host cimport vl_type

cdef extern from "vl/vlad.h":
    void vl_vlad_encode(void* enc,
                        vl_type dataType,
                        void* means,
                        vl_size dimension,
                        vl_size numClusters,
                        void* data,
                        vl_size numData,
                        void * assignments,
                        int flags)


    cdef int VL_VLAD_FLAG_NORMALIZE_COMPONENTS "VL_VLAD_FLAG_NORMALIZE_COMPONENTS"
    cdef int VL_VLAD_FLAG_SQUARE_ROOT "VL_VLAD_FLAG_SQUARE_ROOT"
    cdef int VL_VLAD_FLAG_UNNORMALIZED "VL_VLAD_FLAG_UNNORMALIZED"
    cdef int VL_VLAD_FLAG_NORMALIZE_MASS "VL_VLAD_FLAG_NORMALIZE_MASS"
