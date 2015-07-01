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
from libc.stdlib cimport qsort

# Import the header files
from cyvlfeat._vl.dsift cimport *
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.sift cimport *
from cyvlfeat._vl.mathop cimport VL_PI


@cython.boundscheck(False)
cpdef dsift(np.ndarray[float, ndim=2, mode='c'] data, int[:] step,
            int[:] size, int[:] bounds, int window_size, bint norm, bint fast,
            bint float_descriptors, int[:] geometry, bint verbose):

    cdef:
        int num_frames = 0
        VlDsiftKeypoint *frames_array
        float *descriptors_array
        int k = 0, i = 0
        int step_x = 0, step_y = 0, min_x = 0, min_y = 0, max_x = 0, max_y = 0

        np.ndarray[float, ndim=2, mode='c'] out_descriptors
        np.ndarray[float, ndim=2, mode='c'] out_frames

        int height = data.shape[0]
        int width = data.shape[1]

        VlDsiftDescriptorGeometry geom
        VlDsiftFilter *dsift = vl_dsift_new(width, height)

        int ndims = 0
        int descriptor_index = 0
        float* linear_descriptor

    # Setup the geometry (number of bins and sizes)
    # Note the y-axis is taken as the first (zeroth) axis
    geom.numBinX = geometry[1]
    geom.numBinY = geometry[0]
    geom.numBinT = geometry[2]
    geom.binSizeX = size[1]
    geom.binSizeY = size[0]
    vl_dsift_set_geometry(dsift, &geom)

    # Set other options
    vl_dsift_set_steps(dsift, step[1], step[0])
    vl_dsift_set_bounds(dsift, bounds[1], bounds[0], bounds[3], bounds[2])
    vl_dsift_set_flat_window(dsift, fast)

    if window_size >= 0:
      vl_dsift_set_window_size(dsift, window_size)

    # Get calculated values from the dsift object
    num_frames = vl_dsift_get_keypoint_num(dsift)
    descriptor_length = vl_dsift_get_descriptor_size(dsift)
    geom = deref(vl_dsift_get_geometry(dsift))

    if verbose:
      vl_dsift_get_steps(dsift, &step_x, &step_y)
      vl_dsift_get_bounds(dsift, &min_x, &min_y, &max_x, &max_y)

      printf("vl_dsift: image size         [W, H] = [%d, %d]\n", width, height)
      printf("vl_dsift: bounds:            "
             "[minX, minY, maxX, maxY] = [%d, %d, %d, %d]\n",
             min_x, min_y, max_x, max_y)
      printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n",
             step_x, step_y)
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
    out_descriptors = np.empty((num_frames, descriptor_length),
                               dtype=np.float32, order='C')
    # Grab the pointer to the data so we can walk it linearly
    linear_descriptor = &out_descriptors[0, 0]

    # The norm is added as the third component if set
    if norm:
        ndims = 3
        out_frames = np.empty((num_frames, ndims), dtype=np.float32)
    else:
        ndims = 2
        out_frames = np.empty((num_frames, ndims), dtype=np.float32)

    # Copy results out
    for k in range(num_frames):
        out_frames[k, 0] = frames_array[k].y
        out_frames[k, 1] = frames_array[k].x

        # We have an implied / 2 in the norm, because of the clipping below
        if norm:
            out_frames[k, 2] = frames_array[k].norm

        # We don't need to transpose because our memory is in the correct
        # order already!
        for i in range(descriptor_length):
            descriptor_index = num_frames * i + k
            linear_descriptor[descriptor_index] = \
                min(512.0 * descriptors_array[descriptor_index], 255.0)

    # Clean up the allocated memory
    vl_dsift_delete(dsift)

    if float_descriptors:
        return out_frames, out_descriptors
    else:
        return out_frames, out_descriptors.astype(np.uint8)


cdef int korder(const void *a, const void *b) nogil:
    cdef double x = (<double*> a)[2] - (<double*> b)[2]
    if x < 0: return -1
    if x > 0: return +1
    return 0

cdef inline void transpose_descriptor(vl_sift_pix* dst, vl_sift_pix* src) nogil:
    cdef:
        int BO = 8  # number of orientation bins
        int BP = 4  # number of spatial bins
        int i = 0, j = 0, t = 0, jp = 0, o = 0, op = 0

    for j in range(BP):
        jp = BP - 1 - j
        for i in range(BP):
            o  = BO * i + BP * BO * j
            op = BO * i + BP * BO * jp
            dst [op] = src[o]
            for t in range(1, BO):
                dst [BO - t + op] = src [t + o]


@cython.boundscheck(False)
cpdef sift(np.ndarray[float, ndim=2, mode='fortran'] data, int n_octaves,
           int n_levels, int first_octave, int peak_thresh,
           int edge_thresh, float norm_thresh, int magnif,
           int window_size, float[:, :] frames, bint force_orientations,
           bint float_descriptors, bint compute_descriptor, bint verbose):

    cdef:
        bint first = False
        int nikeys = 0, nframes = 0, reserved = 0, i = 0, j = 0, q = 0
        int height = data.shape[0]
        int width = data.shape[1]
        float *ikeys
        VlSiftFilt *filt = vl_sift_new (height, width, n_octaves,
                                        n_levels, first_octave)

        np.ndarray[float, ndim=2, mode='fortran'] out_descriptors = np.empty((1, 1), dtype=np.float32, order='F')
        np.ndarray[float, ndim=2, mode='c'] out_frames = np.empty((1, 4), dtype=np.float32, order='C')

        float *descr = &out_descriptors[0, 0]
        float *flat_out_frames = &out_frames[0, 0]

        int err = 0
        VlSiftKeypoint *keys
        int nkeys = 0

        double[4] angles
        int nangles = 0
        VlSiftKeypoint ik
        VlSiftKeypoint *k

        vl_sift_pix[128] buf
        vl_sift_pix[128] rbuf

        float x = 0

        bint user_specified_frames = False

    user_specified_frames = frames is not None
    if user_specified_frames:
        nikeys = frames.shape[0]
        ikeys = &frames[0, 0]
        # Ensure frames array is sorted
        qsort(ikeys, nikeys, 4 * sizeof(double), korder)

    if peak_thresh >= 0: vl_sift_set_peak_thresh(filt, peak_thresh)
    if edge_thresh >= 0: vl_sift_set_edge_thresh(filt, edge_thresh)
    if norm_thresh >= 0: vl_sift_set_norm_thresh(filt, norm_thresh)
    if magnif      >= 0: vl_sift_set_magnif(filt, magnif)
    if window_size >= 0: vl_sift_set_window_size(filt, window_size)

    if verbose:
        printf("vl_sift: filter settings:\n")
        printf("vl_sift:   octaves      (O)      = %d\n", vl_sift_get_noctaves(filt))
        printf("vl_sift:   levels       (S)      = %d\n", vl_sift_get_nlevels(filt))
        printf("vl_sift:   first octave (o_min)  = %d\n", vl_sift_get_octave_first(filt))
        printf("vl_sift:   edge thresh           = %g\n", vl_sift_get_edge_thresh(filt))
        printf("vl_sift:   peak thresh           = %g\n", vl_sift_get_peak_thresh(filt))
        printf("vl_sift:   norm thresh           = %g\n", vl_sift_get_norm_thresh(filt))
        printf("vl_sift:   window size           = %g\n", vl_sift_get_window_size(filt))
        printf("vl_sift:   float descriptor      = %d\n", float_descriptors)

        printf("vl_sift: will source frames? yes (%d read)\n"
               if user_specified_frames
               else "vl_sift: will source frames? no\n", nikeys)
        printf("vl_sift: will force orientations? %d\n", force_orientations)

    # Process each octave
    i = 0
    first = True
    while True:
        err = 0
        nkeys = 0

        if verbose:
            printf("vl_sift: processing octave %d\n", vl_sift_get_octave_index(filt))

        # Calculate the GSS for the next octave ....................
        if first:
            err = vl_sift_process_first_octave(filt, &data[0, 0])
            first = False
        else:
            err = vl_sift_process_next_octave(filt)

        if err:
            break

        if verbose:
            printf("vl_sift: GSS octave %d computed\n", vl_sift_get_octave_index(filt))

        # Run detector .............................................
        if not user_specified_frames:
            vl_sift_detect(filt)

            keys = vl_sift_get_keypoints (filt)
            nkeys = vl_sift_get_nkeypoints(filt)
            i = 0

            if verbose:
                printf("vl_sift: detected %d (unoriented) keypoints\n", nkeys)
        else:
            nkeys = nikeys

        # For each keypoint ........................................
        for i in range(nkeys):
            # Obtain keypoint orientations ...........................
            if user_specified_frames:
                vl_sift_keypoint_init (filt, &ik,
                                       ikeys[4 * i + 1] - 1,
                                       ikeys[4 * i + 0] - 1,
                                       ikeys[4 * i + 2])

                if ik.o != vl_sift_get_octave_index(filt):
                    break

                k = &ik

                # optionally compute orientations too
                if force_orientations:
                    nangles = vl_sift_calc_keypoint_orientations(filt, angles, k)
                else:
                    angles[0] = VL_PI / 2 - ikeys [4 * i + 3]
                    nangles    = 1
            else:
                k = keys + i
                nangles = vl_sift_calc_keypoint_orientations(filt, angles, k)

                # For each orientation ...................................
                for q in range(nangles):
                    # compute descriptor (if necessary)
                    if compute_descriptor:
                        vl_sift_calc_keypoint_descriptor(filt, buf, k,
                                                         angles[q])
                        transpose_descriptor(rbuf, buf)

                    # make enough room for all these keypoints and more
                    if reserved < nframes + 1:
                        reserved += 2 * nkeys
                        out_frames = np.resize(out_frames, (reserved, 4))
                        flat_out_frames = &out_frames[0, 0]
                        if compute_descriptor:
                            out_descriptors = np.resize(out_descriptors, (128, reserved))
                            descr = &out_descriptors[0, 0]

                    # Save back with MATLAB conventions. Notice tha the input
                    # image was the transpose of the actual image.
                    flat_out_frames[4 * nframes + 0] = k.y + 1
                    flat_out_frames[4 * nframes + 1] = k.x + 1
                    flat_out_frames[4 * nframes + 2] = k.sigma
                    flat_out_frames[4 * nframes + 3] = VL_PI / 2 - angles[q]

                    if compute_descriptor:
                        if not float_descriptors:
                            for j in range(128):
                                x = min(512.0 * rbuf[j], 255.0)
                                descr[128 * nframes + j] = x

                    nframes += 1

    if verbose:
        printf("vl_sift: found %d keypoints\n", nframes)

    # cleanup
    vl_sift_delete(filt)

    out_frames = np.resize(out_frames, (nframes, 4))

    if compute_descriptor:
        if float_descriptors:
            return out_frames, out_descriptors
        else:
            return out_frames, out_descriptors.astype(np.uint8)
    else:
        return out_frames
