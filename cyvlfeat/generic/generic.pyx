# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
cimport cython
from cyvlfeat._vl.generic cimport *


cpdef void set_simd_enabled(bint x):
    vl_set_simd_enabled(x)

cpdef bint get_simd_enabled():
    return vl_get_simd_enabled()

cpdef bint cpu_has_avx():
    return vl_cpu_has_avx()

cpdef bint cpu_has_sse3():
    return vl_cpu_has_sse3()

cpdef bint cpu_has_sse2():
    return vl_cpu_has_sse2()

cpdef int get_num_cpus():
    return vl_get_num_cpus()

cpdef int get_max_threads():
    return vl_get_max_threads()

cpdef void set_num_threads(int n):
    vl_set_num_threads(n)

cpdef int get_thread_limit():
    return vl_get_thread_limit()


