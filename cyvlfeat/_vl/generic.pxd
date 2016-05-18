# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size, vl_bool

cdef extern from "vl/generic.h":
    void vl_set_simd_enabled(vl_bool x)
    vl_bool vl_get_simd_enabled()
    vl_bool vl_cpu_has_avx()
    vl_bool vl_cpu_has_sse3()
    vl_bool vl_cpu_has_sse2()
    vl_size vl_get_num_cpus()
    
    vl_size vl_get_max_threads()
    void vl_set_num_threads(vl_size n)
    vl_size vl_get_thread_limit()
