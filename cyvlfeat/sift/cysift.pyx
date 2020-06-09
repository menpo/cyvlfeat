# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
from libc.stdlib cimport qsort

# Import the header files
from cyvlfeat._vl.dsift cimport *
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.sift cimport *
from cyvlfeat._vl.mathop cimport VL_PI
from cyvlfeat.cy_util cimport py_printf, set_python_vl_printf


@cython.boundscheck(False)
cpdef cy_dsift(float[:, ::1] data, int[:] step,
               int[:] size, int[:] bounds, int window_size, bint norm,
               bint fast, bint float_descriptors, int[:] geometry,
               bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        int num_frames = 0
        const VlDsiftKeypoint *frames_array
        const float *descriptors_array
        int k = 0, i = 0
        int step_x = 0, step_y = 0, min_x = 0, min_y = 0, max_x = 0, max_y = 0

        float[:, ::1] out_descriptors
        float[:, ::1] out_frames

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

      py_printf("vl_dsift: image size         [W, H] = [%d, %d]\n", width, height)
      py_printf("vl_dsift: bounds:            "
                "[minX, minY, maxX, maxY] = [%d, %d, %d, %d]\n",
                min_x, min_y, max_x, max_y)
      py_printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n",
                step_x, step_y)
      py_printf("vl_dsift: num bins:          "
                "[numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
                geom.numBinT,
                geom.numBinX,
                geom.numBinY)
      py_printf("vl_dsift: descriptor size:   %d\n", descriptor_length)
      py_printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
                geom.binSizeX,
                geom.binSizeY)
      py_printf("vl_dsift: flat window:       %d\n", fast)
      py_printf("vl_dsift: window size:       "
                "%g\n", vl_dsift_get_window_size(dsift))
      py_printf("vl_dsift: num of features:   %d\n", num_frames)

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
        return np.asarray(out_frames), np.asarray(out_descriptors)
    else:
        return np.asarray(out_frames), np.asarray(out_descriptors).astype(np.uint8)


cdef int korder(const void *a, const void *b) nogil:
    cdef float x = (<float*> a)[2] - (<float*> b)[2]
    if x < 0: return -1
    if x > 0: return +1
    return 0


@cython.boundscheck(False)
cpdef cy_sift(float[:, ::1] data, int n_octaves,
              int n_levels, int first_octave, float peak_threshold,
              float edge_threshold, float norm_threshold, int magnification,
              int window_size, float[:, :] frames, bint force_orientations,
              bint float_descriptors, bint compute_descriptor, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        bint is_first_octave = True
        int n_user_keypoints = 0, total_keypoints = 0, reserved = 0, i = 0, j = 0, q = 0
        int height = data.shape[0]
        int width = data.shape[1]
        float *user_keypoints_arr
        VlSiftFilt *filt = vl_sift_new(width, height, n_octaves, n_levels,
                                       first_octave)

        # Create empty 2D output arrays
        float[:, ::1] out_descriptors = np.empty((0, 128), dtype=np.float32,
                                                 order='C')
        float[:, ::1] out_frames = np.empty((0, 4), dtype=np.float32, order='C')

        float *flat_descriptors = &out_descriptors[0, 0]
        float *flat_out_frames = &out_frames[0, 0]

        int is_octaves_complete = 0
        const VlSiftKeypoint *keypoints
        int n_keypoints = 0

        double[4] angles
        int n_angles = 0
        const VlSiftKeypoint *curr_keypoint
        VlSiftKeypoint ik

        vl_sift_pix[128] single_descriptor_arr

        bint user_specified_frames = False

    user_specified_frames = frames is not None
    if user_specified_frames:
        n_user_keypoints = frames.shape[0]
        user_keypoints_arr = &frames[0, 0]
        # Ensure frames array is sorted by increasing scale
        qsort(user_keypoints_arr, n_user_keypoints,
              4 * sizeof(float), korder)

    if peak_threshold  >= 0: vl_sift_set_peak_thresh(filt, peak_threshold)
    if edge_threshold  >= 0: vl_sift_set_edge_thresh(filt, edge_threshold)
    if norm_threshold  >= 0: vl_sift_set_norm_thresh(filt, norm_threshold)
    if magnification   >= 0: vl_sift_set_magnif(filt, magnification)
    if window_size     >= 0: vl_sift_set_window_size(filt, window_size)

    if verbose:
        py_printf("vl_sift: filter settings:\n")
        py_printf("vl_sift:   octaves      (O)      = %d\n", vl_sift_get_noctaves(filt))
        py_printf("vl_sift:   levels       (S)      = %d\n", vl_sift_get_nlevels(filt))
        py_printf("vl_sift:   first octave (o_min)  = %d\n", vl_sift_get_octave_first(filt))
        py_printf("vl_sift:   edge thresh           = %g\n", vl_sift_get_edge_thresh(filt))
        py_printf("vl_sift:   peak thresh           = %g\n", vl_sift_get_peak_thresh(filt))
        py_printf("vl_sift:   norm thresh           = %g\n", vl_sift_get_norm_thresh(filt))
        py_printf("vl_sift:   window size           = %g\n", vl_sift_get_window_size(filt))
        py_printf("vl_sift:   float descriptor      = %d\n", float_descriptors)

        py_printf("vl_sift: will source frames? yes (%d read)\n"
                  if user_specified_frames
                  else "vl_sift: will source frames? no\n", n_user_keypoints)
        py_printf("vl_sift: will force orientations? %d\n", force_orientations)


    if user_specified_frames:
        # If we have specified the frames, we know how many frames
        # will be calculated, so we can skip the dynamic reallocation
        # which is normally done inside the keypoints loop
        out_frames = np.resize(out_frames, (n_user_keypoints, 4))
        flat_out_frames = &out_frames[0, 0]
        # Similar for the descriptors, if necessary
        if compute_descriptor:
            out_descriptors = np.resize(out_descriptors,
                                        (n_user_keypoints, 128))
            flat_descriptors = &out_descriptors[0, 0]

    # Process each octave
    while True:
        if verbose:
            py_printf("vl_sift: processing octave %d\n",
                      vl_sift_get_octave_index(filt))

        # Calculate the GSS for the next octave ....................
        if is_first_octave:
            is_octaves_complete = vl_sift_process_first_octave(filt,
                                                               &data[0, 0])
            is_first_octave = False
        else:
            is_octaves_complete = vl_sift_process_next_octave(filt)

        if is_octaves_complete:
            break

        if verbose:
            py_printf("vl_sift: GSS octave %d computed\n",
                      vl_sift_get_octave_index(filt))

        # Run detector .............................................
        if not user_specified_frames:
            i = 0
            vl_sift_detect(filt)

            keypoints = vl_sift_get_keypoints(filt)
            n_keypoints = vl_sift_get_nkeypoints(filt)

            if verbose:
                py_printf("vl_sift: detected %d (unoriented) keypoints\n",
                          n_keypoints)
        else:
            n_keypoints = n_user_keypoints

        # For each keypoint
        while i < n_keypoints:
            # Obtain keypoint orientations
            if user_specified_frames:
                vl_sift_keypoint_init(filt, &ik,
                                      user_keypoints_arr[4 * i + 1],
                                      user_keypoints_arr[4 * i + 0],
                                      user_keypoints_arr[4 * i + 2])

                if ik.o != vl_sift_get_octave_index(filt):
                    break

                curr_keypoint = &ik

                # Optionally force computation of orientations
                if force_orientations:
                    n_angles = vl_sift_calc_keypoint_orientations(filt, angles,
                                                                  curr_keypoint)
                else:
                    angles[0] = user_keypoints_arr[4 * i + 3]
                    n_angles  = 1
            else:
                # This is equivalent to &keypoints[i] - just get the pointer
                # to the i'th element.
                curr_keypoint = keypoints + i
                n_angles = vl_sift_calc_keypoint_orientations(filt, angles,
                                                              curr_keypoint)

            # For each orientation
            for q in range(n_angles):
                if compute_descriptor:
                    vl_sift_calc_keypoint_descriptor(filt,
                                                     single_descriptor_arr,
                                                     curr_keypoint, angles[q])

                # Dynamically reallocate the output arrays so that they can
                # fit all the keypoints being requested.
                # If statement says: IF we will run out of space next iteration
                #                    AND we have computed the frame OR the user
                #                        has allowed estimation of the number of
                #                        orientations AND there was more than one
                #                    THEN reallocate memory
                if (reserved < total_keypoints + 1 and
                   (not user_specified_frames or
                    (force_orientations and n_angles > 1))):
                    reserved += 2 * n_keypoints

                    out_frames = np.resize(out_frames, (reserved, 4))
                    flat_out_frames = &out_frames[0, 0]

                    if compute_descriptor:
                        out_descriptors = np.resize(out_descriptors,
                                                    (reserved, 128))
                        flat_descriptors = &out_descriptors[0, 0]

                # Notice that this method will give different results
                # from MATLAB because MATLAB actually runs on the
                # transpose of the image due to it's fortran ordering!
                flat_out_frames[total_keypoints * 4 + 0] = curr_keypoint.y
                flat_out_frames[total_keypoints * 4 + 1] = curr_keypoint.x
                flat_out_frames[total_keypoints * 4 + 2] = curr_keypoint.sigma
                flat_out_frames[total_keypoints * 4 + 3] = angles[q]

                if compute_descriptor:
                    for j in range(128):
                        flat_descriptors[total_keypoints * 128 + j] = \
                            min(512.0 * single_descriptor_arr[j], 255.0)

                total_keypoints += 1
            i += 1

    if verbose:
        py_printf("vl_sift: found %d keypoints\n", total_keypoints)

    # cleanup
    vl_sift_delete(filt)

    # If we have dynamically allocated memory for the frames, make sure that
    # we resize the array back to the correct size (since we optimistically
    # allocated previously to reduce the number of total resizes)
    if out_frames.shape[0] != total_keypoints:
        out_frames = np.resize(out_frames, (total_keypoints, 4))
        out_descriptors = np.resize(out_descriptors, (total_keypoints, 128))

    if compute_descriptor:
        if float_descriptors:
            return np.asarray(out_frames), np.asarray(out_descriptors)
        else:
            return np.asarray(out_frames), np.asarray(out_descriptors).astype(np.uint8)
    else:
        return np.asarray(out_frames)
