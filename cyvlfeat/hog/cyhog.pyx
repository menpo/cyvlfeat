# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython

# Import the header files
from cyvlfeat._vl.hog cimport *
from cyvlfeat._vl.host cimport VL_FALSE, vl_size
from cyvlfeat.cy_util cimport set_python_vl_printf


@cython.boundscheck(False)
cpdef cy_hog(float[:, :, ::1] data, vl_size cell_size, int variant,
             vl_size n_orientations, bint directed_polar_field,
             bint undirected_polar_field, bint bilinear_interpolation,
             bint return_channels_last_axis, bint verbose, bint visualize):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        # Python images are not transposed
        # RGB image is channel first, aka. (C, H, W)
        VlHog* hog = vl_hog_new(<VlHogVariant>variant,
                                n_orientations, VL_FALSE)
        vl_size height = data.shape[1]
        vl_size width = data.shape[2]
        vl_size n_channels = data.shape[0]
        vl_size out_height = 0, out_width = 0, out_n_channels = 0
        float[:, :, ::1] out_array
        float[:, ::1] viz_array
        vl_size viz_glyph_size

    vl_hog_set_use_bilinear_orientation_assignments(hog,
                                                    bilinear_interpolation)

    if directed_polar_field or undirected_polar_field:
        vl_hog_put_polar_field(hog,
                               &data[0, 0, 0],  # Magnitude
                               &data[1, 0, 0],  # Angle
                               directed_polar_field,
                               width, height, cell_size)
    else:  # Assume we have an image
        vl_hog_put_image(hog, &data[0, 0, 0], width, height, n_channels,
                         cell_size)

    out_height = vl_hog_get_height(hog)
    out_width = vl_hog_get_width(hog)
    out_n_channels = vl_hog_get_dimension(hog)

    if verbose:
        print('vl_hog: image: [%lld x %lld x %lld]' % (height, width, n_channels))
        print('vl_hog: descriptor: [%lld x %lld x %lld]' % out_height, out_width, out_n_channels)
        print('vl_hog: number of orientations: %lld' % (n_orientations))
        print('vl_hog: bilinear orientation assignments: %s' %
              'yes' if bilinear_interpolation else 'no')
        print('vl_hog: variant: %s' %
              'DalalTriggs' if variant == VlHogVariantDalalTriggs else 'UOCTTI')
        print('vl_hog: input type: %s' %
              'DirectedPolarField' if directed_polar_field
              else ('UndirectedPolarField' if undirected_polar_field else 'Image'))

    # according to https://www.vlfeat.org/api/hog.html
    # hog features array is (out_n_channels, out_height, out_width) with "C" order
    out_array = np.zeros((out_n_channels, out_height, out_width),
                         dtype=np.float32, order='C')

    vl_hog_extract(hog, &out_array[0][0][0])

    if visualize:
        viz_glyph_size = vl_hog_get_glyph_size(hog)
        viz_width = viz_glyph_size * vl_hog_get_width(hog)
        viz_height = viz_glyph_size * vl_hog_get_height(hog)
        viz_array = np.zeros((viz_glyph_size * out_height,
                            viz_glyph_size * out_width),
                            dtype=np.float32, order='C')
        vl_hog_render(hog, &viz_array[0][0], &out_array[0][0][0],
                      out_width, out_height)
        if verbose:
            print("vl_hog: glyph size: %lld" % viz_glyph_size)
            print("vl_hog: glyph image: [%lld x %lld]" %
                  viz_glyph_size * out_height,
                  viz_glyph_size * out_width)

    vl_hog_delete(hog)
    # we prefer (out_height, out_width, out_n_channels) in numpy
    if return_channels_last_axis:
        out_array = np.transpose(out_array, [1, 2, 0]).copy()
    if visualize:
        return np.asarray(out_array), np.asarray(viz_array)
    else:
        return np.asarray(out_array)
