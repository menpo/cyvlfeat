# distutils: include_dirs = cyvlfeat
cimport cython
cimport numpy as np


cdef extern from "cy_util.h":
    void py_printf(const char *format, ...)
    void set_python_vl_printf()


cdef inline np.dtype dtype_from_memoryview(cython.view.memoryview arr):
    return np.dtype(arr.view.format)
