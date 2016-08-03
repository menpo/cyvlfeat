# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
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
    """

    Returns
    -------
        Number of CPU cores.
    """
    return vl_get_num_cpus()

cpdef int get_max_threads():
    """
    This function returns the maximum number of thread used by VLFeat.
    VLFeat will try to use this number of computational threads and never exceed it.

    This is similar to the OpenMP function omp_get_max_threads(); however, it reads
    a parameter private to VLFeat which is independent of the value used by the OpenMP library.

    If VLFeat was compiled without OpenMP support, this function returns 1.

    Returns
    -------
        Number of threads.
    """
    return vl_get_max_threads()

cpdef void set_num_threads(int num_threads):
    """
    This function sets the maximum number of computational threads that will be used by VLFeat.
    VLFeat may in practice use fewer threads (for example because ``num_threads`` is larger than the
    number of computational cores in the host, or because the number of threads exceeds the
    limit available to the application).

    If ``num_threads`` is set to 0, then VLFeat sets the number of threads to the OpenMP current
    maximum, obtained by calling omp_get_max_threads().

    This function is similar to omp_set_num_threads() but changes a parameter internal
    to VLFeat rather than affecting OpenMP global state.

    If VLFeat was compiled without, this function does nothing.
    """
    vl_set_num_threads(num_threads)

cpdef int get_thread_limit():
    """
    This function wraps the OpenMP function omp_get_thread_limit().
    If VLFeat was compiled without OpenMP support, this function returns 1.
    If VLFeat was compiled with OpenMP prior to version 3.0 (2008/05), it returns 0.

    Returns
    -------
        Thread limit.
    """
    return vl_get_thread_limit()
