# Copyright(C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.

from .host cimport vl_size


cdef extern from "vl/sift.h":
    ctypedef float vl_sift_pix

    ctypedef struct VlSiftKeypoint:
        int o          # < o coordinate(octave).
        int ix         # < Integer unnormalized x coordinate.
        int iy         # < Integer unnormalized y coordinate.
        int c_is "is"  # < Integer s coordinate.
        float x        # < x coordinate.
        float y        # < y coordinate.
        float s        # < s coordinate.
        float sigma    # < scale.

    ctypedef struct VlSiftFilt:
        double sigman        # < nominal image smoothing. 
        double sigma0        # < smoothing of pyramid base. 
        double sigmak        # < k-smoothing 
        double dsigma0       # < delta-smoothing.
        int width            # < image width. 
        int height           # < image height. 
        int O                # < number of octaves. 
        int S                # < number of levels per octave. 
        int o_min            # < minimum octave index. 
        int s_min            # < minimum level index. 
        int s_max            # < maximum level index. 
        int o_cur            # < current octave.
        vl_sift_pix *temp    # < temporary pixel buffer. 
        vl_sift_pix *octave  # < current GSS data. 
        vl_sift_pix *dog     # < current DoG data. 
        int octave_width     # < current octave width. 
        int octave_height    # < current octave height.
        vl_sift_pix *gaussFilter   # < current Gaussian filter 
        double gaussFilterSigma    # < current Gaussian filter std 
        vl_size gaussFilterWidth   # < current Gaussian filter width
        VlSiftKeypoint* keys # < detected keypoints. 
        int nkeys            # < number of detected keypoints. 
        int keys_res         # < size of the keys buffer.
        double peak_thresh   # < peak threshold. 
        double edge_thresh   # < edge threshold. 
        double norm_thresh   # < norm threshold. 
        double magnif        # < magnification factor. 
        double windowSize    # < size of Gaussian window(in spatial bins)
        vl_sift_pix *grad    # < GSS gradient data. 
        int grad_o           # < GSS gradient data octave.

    VlSiftFilt* vl_sift_new(int width, int height, int noctaves, int nlevels,
                            int o_min)
    void vl_sift_delete(VlSiftFilt *f) 
    int vl_sift_process_first_octave(VlSiftFilt *f, vl_sift_pix *im)
    int vl_sift_process_next_octave(VlSiftFilt *f) 
    void vl_sift_detect(VlSiftFilt *f) 
    int vl_sift_calc_keypoint_orientations(VlSiftFilt *f, double *angles,
                                            VlSiftKeypoint *k)
    void vl_sift_calc_keypoint_descriptor(VlSiftFilt *f, vl_sift_pix *descr,
                                          VlSiftKeypoint * k, double angle)
    void vl_sift_calc_raw_descriptor(VlSiftFilt *f,
                                     vl_sift_pix * image,
                                     vl_sift_pix *descr, int width, int height,
                                     double x, double y, double s,
                                     double angle0)
    void vl_sift_keypoint_init(VlSiftFilt *f, VlSiftKeypoint *k, double x,
                               double y, double sigma)
    inline int vl_sift_get_octave_index(VlSiftFilt *f)
    inline int vl_sift_get_noctaves(VlSiftFilt *f)
    inline int vl_sift_get_octave_first(VlSiftFilt *f)
    inline int vl_sift_get_octave_width(VlSiftFilt *f)
    inline int vl_sift_get_octave_height(VlSiftFilt *f)
    inline int vl_sift_get_nlevels(VlSiftFilt *f)
    inline int vl_sift_get_nkeypoints(VlSiftFilt *f)
    inline double vl_sift_get_peak_thresh(VlSiftFilt *f)
    inline double vl_sift_get_edge_thresh(VlSiftFilt *f)
    inline double vl_sift_get_norm_thresh(VlSiftFilt *f)
    inline double vl_sift_get_magnif(VlSiftFilt *f)
    inline double vl_sift_get_window_size(VlSiftFilt *f)
    inline vl_sift_pix *vl_sift_get_octave  (VlSiftFilt *f, int s)
    inline VlSiftKeypoint *vl_sift_get_keypoints (VlSiftFilt *f)
    inline void vl_sift_set_peak_thresh(VlSiftFilt *f, double t) 
    inline void vl_sift_set_edge_thresh(VlSiftFilt *f, double t) 
    inline void vl_sift_set_norm_thresh(VlSiftFilt *f, double t) 
    inline void vl_sift_set_magnif(VlSiftFilt *f, double m) 
    inline void vl_sift_set_window_size(VlSiftFilt *f, double m) 
