# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
from cython.operator cimport dereference as deref
cimport cython
cimport numpy as np


cdef extern from "stdio.h":
  printf(char* string, ...)

cdef extern from "../../vlfeat/vl/host.h":
    ctypedef int vl_bool
    ctypedef unsigned char vl_uint8
    ctypedef unsigned long long vl_size
    ctypedef unsigned long long vl_uindex
    cdef enum:
        VL_FALSE = 0
    cdef enum:
        VL_TRUE = 1

cdef extern from "../../vlfeat/vl/sift.h":
    ctypedef float vl_sift_pix

cdef extern from "../../vlfeat/vl/dsift.h":
    ctypedef struct VlDsiftKeypoint:
        double x  #< x coordinate */
        double y  #< y coordinate */
        double s  #< scale */
        double norm  #< SIFT descriptor norm */

# @brief Dense SIFT descriptor geometry */
    ctypedef struct VlDsiftDescriptorGeometry:
        int numBinT   #< number of orientation bins */
        int numBinX   #< number of bins along X */
        int numBinY   #< number of bins along Y */
        int binSizeX  #< size of bins along X */
        int binSizeY  #< size of bins along Y */


    ctypedef struct VlDsiftFilter:
        int imWidth             #< @internal @brief image width */
        int imHeight            #< @internal @brief image height */
        
        int stepX               #< frame sampling step X */
        int stepY               #< frame sampling step Y */
        
        int boundMinX           #< frame bounding box min X */
        int boundMinY           #< frame bounding box min Y */
        int boundMaxX           #< frame bounding box max X */
        int boundMaxY           #< frame bounding box max Y */
        
        # descriptor parameters */
        VlDsiftDescriptorGeometry geom 
        
        int useFlatWindow       #< flag: whether to approximate the Gaussian window with a flat one */
        double windowSize       #< size of the Gaussian window */
        
        int numFrames           #< number of sampled frames */
        int descrSize           #< size of a descriptor */
        VlDsiftKeypoint *frames  #< frame buffer */
        float *descrs           #< descriptor buffer */
        
        int numBinAlloc         #< buffer allocated: descriptor size */
        int numFrameAlloc       #< buffer allocated: number of frames  */
        int numGradAlloc        #< buffer allocated: number of orientations */
        
        float **grads           #< gradient buffer */
        float *convTmp1         #< temporary buffer */
        float *convTmp2         #< temporary buffer */
    
    
    VlDsiftFilter *vl_dsift_new(int width, int height)
    VlDsiftFilter *vl_dsift_new_basic(int width, int height, int step,
                                      int binSize)
    void vl_dsift_delete(VlDsiftFilter *self)
    void vl_dsift_process(VlDsiftFilter *self, float*im)
    inline void vl_dsift_transpose_descriptor(float*dst,
                                              float*src,
                                              int numBinT,
                                              int numBinX,
                                              int numBinY)

    inline void vl_dsift_set_steps(VlDsiftFilter *self,
                                   int stepX,
                                   int stepY)
    inline void vl_dsift_set_bounds(VlDsiftFilter *self,
                                    int minX,
                                    int minY,
                                    int maxX,
                                    int maxY)
    inline void vl_dsift_set_geometry(VlDsiftFilter *self,
                                      VlDsiftDescriptorGeometry*geom)
    inline void vl_dsift_set_flat_window(VlDsiftFilter *self,
                                         vl_bool useFlatWindow)
    inline void vl_dsift_set_window_size(VlDsiftFilter *self, double windowSize)

    inline float *vl_dsift_get_descriptors(VlDsiftFilter *self)
    inline int             vl_dsift_get_descriptor_size(VlDsiftFilter *self)
    inline int             vl_dsift_get_keypoint_num(VlDsiftFilter *self)
    inline VlDsiftKeypoint *vl_dsift_get_keypoints(VlDsiftFilter *self)
    inline void            vl_dsift_get_bounds(VlDsiftFilter *self,
                                               int*minX,
                                               int*minY,
                                               int*maxX,
                                               int*maxY)
    inline void            vl_dsift_get_steps(VlDsiftFilter*self,
                                              int*stepX,
                                              int*stepY)
    inline VlDsiftDescriptorGeometry* vl_dsift_get_geometry	(	VlDsiftFilter* 	self	)
    inline vl_bool         vl_dsift_get_flat_window(VlDsiftFilter *self)
    inline double          vl_dsift_get_window_size(VlDsiftFilter *self)

    void _vl_dsift_update_buffers(VlDsiftFilter *self)


cdef inline void transpose_descriptor(vl_sift_pix* dst, vl_sift_pix* src):
    cdef int BO = 8  # number of orientation bins
    cdef int BP = 4  # number of spatial bins
    cdef int i, j, t, jp, o, op

    for j in range(BP):
        jp = BP - 1 - j
        for i in range(BP):
            o = BO * i + BP * BO * j
            op = BO * i + BP * BO * jp
            dst[op] = src[o]
            for t in range(1, BO):
                dst[BO - t + op] = src[t + o]

cdef int korder (void* a, void* b):
    """
    Ordering of tuples by increasing scale
    """
    cdef double x = (<double*> a) [2] - (<double*> b) [2]
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


cdef vl_bool check_sorted(double*keys, vl_size nkeys):
    cdef vl_uindex k
    for k in range(nkeys - 1):
        if korder(keys, keys + 4) > 0:
            return VL_FALSE
        keys += 4
    return VL_TRUE


cpdef dsift(np.ndarray[float, ndim=2, mode='c'] data, int[:] step, int[:] size, int[:] bounds,
            int window_size, bint norm, bint fast, bint float_descriptors,
            int[:] geometry, bint verbose):

    cdef int numFrames = 0
    cdef int descrSize = 0
    cdef VlDsiftKeypoint* frames
    cdef float* descrs
    cdef int k, i = 0
    cdef bint useFlatWindow = fast
    cdef int windowSize = window_size
    cdef int stepX, stepY, minX, minY, maxX, maxY = 0

    cdef float[:] out_descriptors
    cdef double[:] out_frames

    cdef int M = data.shape[0]
    cdef int N = data.shape[1]

    cdef VlDsiftDescriptorGeometry geom
    cdef VlDsiftFilter* dsift = vl_dsift_new(M, N)

    cdef int frame_counter = 0
    cdef int descr_counter = 0

    vl_dsift_set_geometry(dsift, &geom)
    vl_dsift_set_steps(dsift, step[0], step[1])

    if not bounds is None:
      vl_dsift_set_bounds(dsift,
                          max(bounds[1], 0),
                          max(bounds[0], 0),
                          min(bounds[3], M),
                          min(bounds[2], N))

    vl_dsift_set_flat_window(dsift, useFlatWindow) 

    if windowSize >= 0:
      vl_dsift_set_window_size(dsift, windowSize) 


    numFrames = vl_dsift_get_keypoint_num (dsift) 
    descrSize = vl_dsift_get_descriptor_size (dsift) 
    geom = deref(vl_dsift_get_geometry(dsift))

    if verbose:
      vl_dsift_get_steps (dsift, &stepY, &stepX) 
      vl_dsift_get_bounds (dsift, &minY, &minX, &maxY, &maxX) 
      useFlatWindow = vl_dsift_get_flat_window(dsift) 

      printf("vl_dsift: image size         [W, H] = [%d, %d]\n",  N, M)
      printf("vl_dsift: bounds:            [minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
                minX+1, minY+1, maxX+1, maxY+1) 
      printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n", stepX, stepY)
      printf("vl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
                geom.numBinT,
                geom.numBinX,
                geom.numBinY) 
      printf("vl_dsift: descriptor size:   %d\n", descrSize)
      printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
                geom.binSizeX,
                geom.binSizeY)
      # printf("vl_dsift: flat window:       %s\n", VL_YESNO(useFlatWindow))
      printf("vl_dsift: window size:       %g\n", vl_dsift_get_window_size(dsift))
      printf("vl_dsift: num of features:   %d\n", numFrames)


    vl_dsift_process (dsift, &data[0, 0])

    frames = vl_dsift_get_keypoints (dsift) 
    descrs = vl_dsift_get_descriptors (dsift)

    # Create output arrays
    # TODO: if not float_descriptors then should be uint8
    out_descriptors = np.zeros(descrSize * numFrames, dtype=np.float32)
    if norm:
        out_frames = np.zeros(3 * numFrames, dtype=np.float64)
    else:
        out_frames = np.zeros(2 * numFrames, dtype=np.float64)

    # Copy results out
    for k in range(numFrames):
        out_frames[frame_counter] = frames[k].y
        frame_counter += 1
        out_frames[frame_counter] = frames[k].x
        frame_counter += 1

        # We have an implied / 2 in the norm, because of the clipping
        #   below
        if norm:
            out_frames[frame_counter] = frames[k].norm
            frame_counter += 1

        if float_descriptors:
            for i in range(descrSize):
                out_descriptors[descr_counter] = min(512.0 * descrs[i], 255.0)
                descr_counter += 1
        else:
            for i in range(descrSize):
                out_descriptors[descr_counter] = <vl_uint8> min(512.0 * descrs[i],
                                                             255.0)
                descr_counter += 1
    vl_dsift_delete (dsift)

    return np.asarray(out_frames), np.asarray(out_descriptors)
