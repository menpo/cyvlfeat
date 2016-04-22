# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
from .host cimport vl_size

cdef extern from "vl/mathop.h":
    double VL_PI
    ctypedef float (*VlFloatVectorComparisonFunction)(vl_size dimension, const float* X, const float* Y)
    ctypedef double (*VlDoubleVectorComparisonFunction)(vl_size dimension, const double* X, const double* Y)
    
    cdef enum _VlVectorComparisonType:
        VlDistanceL1 = 0,        # l1 distance (squared intersection metric)
        VlDistanceL2 = 1,        # squared l2 distance
        VlDistanceChi2 = 2,      # squared Chi2 distance
        VlDistanceHellinger = 3, # squared Hellinger's distance
        VlDistanceJS = 4,        # squared Jensen-Shannon distance
        VlDistanceMahalanobis = 5,     # squared mahalanobis distance
        VlKernelL1 = 6,          # intersection kernel
        VlKernelL2 = 7,          # l2 kernel
        VlKernelChi2 = 8,        # Chi2 kernel
        VlKernelHellinger = 9,   # Hellinger's kernel
        VlKernelJS = 10          # Jensen-Shannon kernel
    ctypedef _VlVectorComparisonType VlVectorComparisonType

    inline char* vl_get_vector_comparison_type_name(int type)