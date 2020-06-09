# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size, vl_uint32, vl_uint8, vl_int32, vl_uint


cdef extern from "vl/ikmeans.h":
    ctypedef vl_int32 vl_ikmacc_t

    cdef enum VlIKMAlgorithms:
        VL_IKM_LLOYD = 0,  # Lloyd algorithm
        VL_IKM_ELKAN = 1   # Elkan algorithm

    cdef struct _VlIKMFilt:
      vl_size M                # data dimensionality
      vl_size K                # number of centers
      vl_size max_niters       # Lloyd: maximum number of iterations
      int method               # Learning method
      int verb                 # verbosity level
      vl_ikmacc_t *centers     # centers
      vl_ikmacc_t *inter_dist  # centers inter-distances
    ctypedef _VlIKMFilt VlIKMFilt
    
    # Create and destroy
    VlIKMFilt* vl_ikm_new (int method)
    void vl_ikm_delete (VlIKMFilt *f)
    
    # Process data
    void vl_ikm_init (VlIKMFilt *f, vl_ikmacc_t *centers, vl_size M, vl_size K)
    void vl_ikm_init_rand (VlIKMFilt *f, vl_size M, vl_size K)
    void vl_ikm_init_rand_data (VlIKMFilt *f, vl_uint8 *data, vl_size M,
                                vl_size N, vl_size K)
    int  vl_ikm_train (VlIKMFilt *f, vl_uint8 *data, vl_size N)
    void vl_ikm_push (VlIKMFilt *f, vl_uint32 *asgn, vl_uint8 *data, vl_size N)
    vl_uint vl_ikm_push_one (vl_ikmacc_t* centers, vl_uint8 *data, vl_size M,
                             vl_size K)

    
    # Retrieve data and parameters
    vl_size vl_ikm_get_ndims (VlIKMFilt *f)
    vl_size vl_ikm_get_K (VlIKMFilt *f)
    int vl_ikm_get_verbosity (VlIKMFilt *f)
    vl_size vl_ikm_get_max_niters (VlIKMFilt *f)
    vl_ikmacc_t* vl_ikm_get_centers (VlIKMFilt *f)
    
    # Set parameters
    void vl_ikm_set_verbosity (VlIKMFilt *f, int verb)
    void vl_ikm_set_max_niters (VlIKMFilt *f, vl_size max_niters)
