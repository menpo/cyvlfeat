# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

from .host cimport vl_size, vl_uint32, vl_uint8, vl_int32, vl_uint, vl_bool

cdef extern from "vl/lbp.h":
    ctypedef enum _VlLbpMappingType:
          VlLbpUniform = 0    # Uniform local binary patterns.
    ctypedef _VlLbpMappingType VlLbpMappingType

    ctypedef struct VlLbp_:
      vl_size dimension
      vl_uint8 mapping [256]
      vl_bool transposed
    ctypedef VlLbp_ VlLbp

    VlLbp * vl_lbp_new (VlLbpMappingType type, vl_bool transposed)
    void vl_lbp_delete(VlLbp * self)
    void vl_lbp_process(VlLbp * self,
                              float * features,
                              float * image, vl_size width, vl_size height,
                              vl_size cellSize)
    vl_size vl_lbp_get_dimension(VlLbp * self

