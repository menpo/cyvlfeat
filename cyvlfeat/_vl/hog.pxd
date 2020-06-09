# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
from .host cimport vl_size, vl_index, vl_bool


cdef extern from "vl/hog.h":
    cdef enum VlHogVariant_:
        VlHogVariantDalalTriggs = 0,
        VlHogVariantUoctti = 1
    ctypedef VlHogVariant_ VlHogVariant

    cdef struct VlHog_:
        VlHogVariant variant
        vl_size dimension
        vl_size numOrientations
        vl_bool transposed
        vl_bool useBilinearOrientationAssigment

        # left-right flip permutation
        vl_index* permutation

        # glyphs
        float* glyphs
        vl_size glyphSize

        # helper vectors
        float* orientationX
        float* orientationY

        # buffers
        float* hog
        float* hogNorm
        vl_size hogWidth
        vl_size hogHeight
    ctypedef VlHog_ VlHog

    VlHog* vl_hog_new(VlHogVariant variant, vl_size numOrientations,
                      vl_bool transposed)
    void vl_hog_delete(VlHog* self)
    void vl_hog_process(VlHog* self, float* features, float* image,
                        vl_size width, vl_size height, vl_size numChannels,
                        vl_size cellSize)

    void vl_hog_put_image(VlHog * self, float* image, vl_size width,
                          vl_size height, vl_size numChannels, vl_size cellSize)

    void vl_hog_put_polar_field(VlHog* self, float* modulus,
                                float* angle, vl_bool directed,
                                vl_size width, vl_size height, vl_size cellSize)

    void vl_hog_extract(VlHog * self, float* features)
    vl_size vl_hog_get_height(VlHog * self)
    vl_size vl_hog_get_width(VlHog * self)


    void vl_hog_render(VlHog * self, float* image, float* features,
                       vl_size width, vl_size height)

    vl_size vl_hog_get_dimension(VlHog* self)
    vl_index* vl_hog_get_permutation(VlHog* self)
    vl_size vl_hog_get_glyph_size(VlHog* self)

    vl_bool vl_hog_get_use_bilinear_orientation_assignments(VlHog* self)
    void vl_hog_set_use_bilinear_orientation_assignments(VlHog* self,
                                                         vl_bool x)
