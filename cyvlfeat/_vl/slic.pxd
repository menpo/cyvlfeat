# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

from cyvlfeat._vl.host cimport vl_size
from cyvlfeat._vl.host cimport vl_uint32

cdef extern from "vl/slic.h":
    void vl_slic_segment(vl_uint32 * segmentation,
                             float * image,
                             vl_size width,
                             vl_size height,
                             vl_size numChannels,
                             vl_size regionSize,
                             float regularization,
                             vl_size minRegionSize)