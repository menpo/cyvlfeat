# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size, vl_uint32, vl_uint8, vl_int32, vl_uint
from .ikmeans cimport VlIKMFilt

cdef extern from "vl/hikmeans.h":
    cdef struct _VlHIKMNode:
        VlIKMFilt* filter  # IKM filter for this node*/
        _VlHIKMNode** children  # Node children (if any)
    ctypedef _VlHIKMNode VlHIKMNode

    cdef struct _VlHIKMTree:
        vl_size M  # IKM: data dimensionality
        vl_size K  # IKM: K
        vl_size depth  # Depth of the tree
        vl_size max_niters   # IKM: maximum # of iterations
        int method  # IKM: method
        int verb  # Verbosity level
        VlHIKMNode * root  # Tree root node
    ctypedef _VlHIKMTree VlHIKMTree

    VlHIKMTree *vl_hikm_new (int method)
    void vl_hikm_delete (VlHIKMTree *f)

    vl_size vl_hikm_get_ndims (VlHIKMTree *f)
    vl_size vl_hikm_get_K (VlHIKMTree *f)
    vl_size vl_hikm_get_depth (VlHIKMTree *f)
    int vl_hikm_get_verbosity (VlHIKMTree *f)
    vl_size vl_hikm_get_max_niters (VlHIKMTree *f)
    VlHIKMNode* vl_hikm_get_root (VlHIKMTree *f)

    void vl_hikm_set_verbosity (VlHIKMTree *f, int verb)
    void vl_hikm_set_max_niters (VlHIKMTree *f, int max_niters)

    void vl_hikm_init (VlHIKMTree *f, vl_size M, vl_size K, vl_size depth)
    void vl_hikm_train (VlHIKMTree *f, vl_uint8 *data, vl_size N)
    void vl_hikm_push (VlHIKMTree *f, vl_uint32 *asgn, vl_uint8 *data, vl_size N)