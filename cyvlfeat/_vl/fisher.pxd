# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size
from .host cimport vl_type


cdef extern from "vl/fisher.h":
    vl_size vl_fisher_encode(void * enc,
                             vl_type dataType,
                             void const * means,
                             vl_size dimension,
                             vl_size numClusters,
                             void const * covariances,
                             void const * priors,
                             void const * data,
                             vl_size numData,
                             int flags)
