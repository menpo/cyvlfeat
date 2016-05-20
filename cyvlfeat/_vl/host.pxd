# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

cdef extern from "vl/host.h":
    ctypedef unsigned int vl_type
    ctypedef int vl_bool
    ctypedef long long vl_int64  # @brief Signed 64-bit integer.
    ctypedef int vl_int32  # @brief Signed 32-bit integer.
    ctypedef short vl_int16  # @brief Signed 16-bit integer.
    ctypedef char vl_int8  # @brief Signed  8-bit integer.
    ctypedef unsigned long long vl_uint64  # @brief Unsigned 64-bit integer.
    ctypedef unsigned int vl_uint32  # @brief Unsigned 32-bit integer.
    ctypedef unsigned short vl_uint16  # @brief Unsigned 16-bit integer.
    ctypedef unsigned char vl_uint8  # @brief Unsigned  8-bit integer.
    ctypedef int vl_int
    ctypedef unsigned int vl_uint
    ctypedef unsigned long long vl_size
    ctypedef unsigned long long vl_uindex
    ctypedef long long vl_index
    cdef int VL_TYPE_FLOAT "VL_TYPE_FLOAT"
    cdef int VL_TYPE_DOUBLE "VL_TYPE_DOUBLE"
    inline char* vl_get_type_name(vl_type type)
    inline vl_size vl_get_type_size(vl_type type)
    cdef enum:
        VL_FALSE = 0
    cdef enum:
        VL_TRUE = 1
