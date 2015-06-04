# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
from libc.stdio cimport printf

# Import the header files
from cyvlfeat._vl.dsift cimport *
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.sift cimport *


@cython.boundscheck(False)
cpdef dsift(np.ndarray[float, ndim=2, mode='fortran'] data, int[:] step,
            int[:] size, int[:] bounds, int window_size, bint norm, bint fast,
            bint float_descriptors, int[:] geometry, bint verbose):

    cdef int num_frames = 0
    cdef int descriptor_length = 0
    cdef VlDsiftKeypoint* frames_array
    cdef float* descriptors_array
    cdef int k, i = 0
    cdef int step_x, step_y, min_x, min_y, max_x, max_y = 0

    cdef np.ndarray[float, ndim=2, mode='fortran'] out_descriptors
    cdef np.ndarray[double, ndim=2, mode='c'] out_frames
    cdef np.ndarray[float, ndim=1, mode='fortran'] single_descriptor_array

    cdef int height = data.shape[0]
    cdef int width = data.shape[1]

    cdef VlDsiftDescriptorGeometry geom
    # Note that height, width does not match the vlfeat annotations, switched
    # for fortran ordering
    cdef VlDsiftFilter* dsift = vl_dsift_new(height, width)

    cdef int ndims = 0
    cdef int descriptor_count = 0
    cdef float* linear_descriptor

    # Setup the geometry (number of bins and sizes)
    # Note the y-axis is taken as the first access but vlfeat expects the x-axis
    # as the first axis
    geom.numBinX = geometry[1]
    geom.numBinY = geometry[0]
    geom.numBinT = geometry[2]
    geom.binSizeX = size[1]
    geom.binSizeY = size[0]
    vl_dsift_set_geometry(dsift, &geom)

    # Set other options
    vl_dsift_set_steps(dsift, step[1], step[0])
    vl_dsift_set_bounds(dsift, bounds[0], bounds[1], bounds[2], bounds[3])
    vl_dsift_set_flat_window(dsift, fast)

    if window_size >= 0:
      vl_dsift_set_window_size(dsift, window_size)

    # Get calculated values from the dsift object
    num_frames = vl_dsift_get_keypoint_num(dsift)
    descriptor_length = vl_dsift_get_descriptor_size(dsift)
    geom = deref(vl_dsift_get_geometry(dsift))

    if verbose:
      vl_dsift_get_steps(dsift, &step_x, &step_x)
      vl_dsift_get_bounds(dsift, &min_x, &min_y, &max_x, &max_y)

      printf("vl_dsift: image size         [W, H] = [%d, %d]\n",  width, height)
      printf("vl_dsift: bounds:            "
             "[minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
             min_x, min_y, max_x, max_y)
      printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n", step_x, step_x)
      printf("vl_dsift: num bins:          "
             "[numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
             geom.numBinT,
             geom.numBinX,
             geom.numBinY)
      printf("vl_dsift: descriptor size:   %d\n", descriptor_length)
      printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
                geom.binSizeX,
                geom.binSizeY)
      printf("vl_dsift: flat window:       %d\n", fast)
      printf("vl_dsift: window size:       "
             "%g\n", vl_dsift_get_window_size(dsift))
      printf("vl_dsift: num of features:   %d\n", num_frames)


    # Actually compute the SIFT features
    vl_dsift_process(dsift, &data[0, 0])

    # Grab the results
    frames_array = vl_dsift_get_keypoints(dsift)
    descriptors_array = vl_dsift_get_descriptors(dsift)

    # Create output arrays
    out_descriptors = np.empty((descriptor_length, num_frames),
                               dtype=np.float32, order='F')
    # Get a temporary array for each descriptor
    single_descriptor_array = np.empty(descriptor_length,
                                       order='F', dtype=np.float32)
    # Grab the pointer to the data so we can walk it linearly
    linear_descriptor = &out_descriptors[0, 0]

    # The norm is added as the third component if set
    if norm:
        ndims = 3
        out_frames = np.empty((ndims, num_frames), dtype=np.float64)
    else:
        ndims = 2
        out_frames = np.empty((ndims, num_frames), dtype=np.float64)

    # Copy results out
    # This is slightly complicated, it follows the Matlab conventions because
    # of the horror that is converting between F and C contiguous arrays
    # with linear indexing.
    for k in range(num_frames):
        out_frames[0, k] = frames_array[k].x
        out_frames[1, k] = frames_array[k].y

        # We have an implied / 2 in the norm, because of the clipping
        #   below
        if norm:
            out_frames[2, k] = frames_array[k].norm

        vl_dsift_transpose_descriptor(&single_descriptor_array[0],
                                      descriptors_array + descriptor_length * k,
                                      geom.numBinT,
                                      geom.numBinX,
                                      geom.numBinY)

        if float_descriptors:
            for i in range(descriptor_length):
                linear_descriptor[descriptor_count] = \
                    min(512.0 * single_descriptor_array[i], 255.0)
                descriptor_count += 1
        else:
            for i in range(descriptor_length):
                linear_descriptor[descriptor_count] = <vl_uint8> \
                    min(512.0 * single_descriptor_array[i], 255.0)
                descriptor_count += 1

    # Clean up the allocated memory
    vl_dsift_delete(dsift)

    return out_frames, out_descriptors
