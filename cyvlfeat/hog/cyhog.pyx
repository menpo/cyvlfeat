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
from cyvlfeat._vl.host cimport VL_FALSE
from cyvlfeat.cy_util cimport py_printf, set_python_vl_printf


@cython.boundscheck(False)
cpdef cy_hog(float[:, :, ::1] data, int cell_size, int variant,
             int n_orientations, bint directed_polar_field,
             bint undirected_polar_field, bint bilinear_interpolation,
             bint return_channels_last_axis, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        # Python images are not transposed
        VlHog* hog = vl_hog_new(<VlHogVariant>variant,
                                n_orientations, VL_FALSE)
        int height = data.shape[0]
        int width = data.shape[1]
        int n_channels = data.shape[2]
        int out_height = 0, out_width = 0, out_n_channels = 0
        float[:, :, :] out_array

    vl_hog_set_use_bilinear_orientation_assignments(hog,
                                                    bilinear_interpolation)

    if directed_polar_field or undirected_polar_field:
        vl_hog_put_polar_field(hog,
                               &data[0, 0, 0],  # Magnitude
                               &data[0, 0, 1],  # Angle
                               directed_polar_field,
                               width, height, cell_size)
    else:  # Assume we have an image
        vl_hog_put_image(hog, &data[0, 0, 0], width, height, n_channels,
                         cell_size)

    out_height = vl_hog_get_height(hog)
    out_width = vl_hog_get_width(hog)
    out_n_channels = vl_hog_get_dimension(hog)

    if verbose:
        py_printf('vl_hog: image: [%d x %d x %d]\n', height, width, n_channels)
        py_printf('vl_hog: descriptor: [%d x %d x %d]\n', out_height, out_width,
                                                       out_n_channels)
        py_printf('vl_hog: number of orientations: %d\n', n_orientations)
        py_printf('vl_hog: bilinear orientation assignments: %s\n',
                  'yes' if bilinear_interpolation else 'no')
        py_printf('vl_hog: variant: %s\n',
                  'DalalTriggs' if variant == VlHogVariantDalalTriggs
                  else 'UOCTTI')
        py_printf('vl_hog: input type: %s\n',
                  'DirectedPolarField' if directed_polar_field
                  else ('UndirectedPolarField' if undirected_polar_field else
                        'Image'))

    # Unfortunately, writing in C-contiguous ordering implies the channels
    # should be at the front.
    out_array = np.empty((out_n_channels, out_height, out_width),
                         dtype=np.float32, order='C')

    vl_hog_extract(hog, &out_array[0, 0, 0])
    vl_hog_delete(hog)

    # Therefore, a copy is required if the channels should be returned as the
    # last axis.
    if return_channels_last_axis:
        out_array = np.transpose(out_array, [1, 2, 0]).copy()

    return np.asarray(out_array)
