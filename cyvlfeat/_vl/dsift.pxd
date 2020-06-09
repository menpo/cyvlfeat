# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
from cyvlfeat._vl.host cimport vl_bool

cdef extern from "vl/dsift.h":
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
