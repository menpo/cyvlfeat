# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size, vl_bool


cdef extern from "vl/quickshift.h":
    ctypedef double vl_qs_type

    ctypedef struct VlQS:
        vl_qs_type *image    # height x width x channels feature image
        int height           # height of the image
        int width            # width of the image
        int channels         # number of channels in the image
        vl_bool medoid
        vl_qs_type sigma
        vl_qs_type tau
        int *parents
        vl_qs_type *dists
        vl_qs_type *density

    VlQS *vl_quickshift_new(vl_qs_type *im, int height, int width, int channels)
    void vl_quickshift_delete(VlQS *q)
    void vl_quickshift_process (VlQS *q)

    inline vl_qs_type vl_quickshift_get_max_dist(VlQS *q)
    inline vl_qs_type vl_quickshift_get_kernel_size(VlQS *q)
    inline vl_bool vl_quickshift_get_medoid(VlQS *q)
    inline int *vl_quickshift_get_parents(VlQS *q)
    inline vl_qs_type *vl_quickshift_get_dists(VlQS *q)
    inline vl_qs_type *vl_quickshift_get_density(VlQS *q)
    inline void vl_quickshift_set_max_dist(VlQS *f, vl_qs_type tau)
    inline void vl_quickshift_set_kernel_size(VlQS *f, vl_qs_type sigma)
    inline void vl_quickshift_set_medoid(VlQS *f, vl_bool medoid)

