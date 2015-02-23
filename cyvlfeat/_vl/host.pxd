# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

cdef extern from "vl/host.h":
    ctypedef int vl_bool
    ctypedef unsigned char vl_uint8
    ctypedef unsigned long long vl_size
    ctypedef unsigned long long vl_uindex
    cdef enum:
        VL_FALSE = 0
    cdef enum:
        VL_TRUE = 1
