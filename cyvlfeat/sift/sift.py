import numpy as np
import cy_sift


def sift(image, octaves=-1, levels=3, first_octave=0,
         peak_thresh=0, edge_thresh=10, norm_thresh = -np.inf,
         magnif = 3, window_size=2, force_orientations=False,
         float_descriptors=False, verbose=0, frames=None):
    """ VL_SIFT  Scale-Invariant Feature Transform
%   F = VL_SIFT(I) computes the SIFT frames [1] (keypoints) F of the
%   image I. I is a gray-scale image in single precision. Each column
%   of F is a feature frame and has the format [X;Y;S;TH], where X,Y
%   is the (fractional) center of the frame, S is the scale and TH is
%   the orientation (in radians).
%
%   [F,D] = VL_SIFT(I) computes the SIFT descriptors [1] as well. Each
%   column of D is the descriptor of the corresponding frame in F. A
%   descriptor is a 128-dimensional vector of class UINT8.
%
%   VL_SIFT() accepts the following options:
%
%   Octaves:: maximum possible
%     Set the number of octave of the DoG scale space.
%
%   Levels:: 3
%     Set the number of levels per octave of the DoG scale space.
%
%   FirstOctave:: 0
%     Set the index of the first octave of the DoG scale space.
%
%   PeakThresh:: 0
%     Set the peak selection threshold.
%
%   EdgeThresh:: 10
%     Set the non-edge selection threshold.
%
%   NormThresh:: -inf
%     Set the minimum l2-norm of the descriptors before
%     normalization. Descriptors below the threshold are set to zero.
%
%   Magnif:: 3
%     Set the descriptor magnification factor. The scale of the
%     keypoint is multiplied by this factor to obtain the width (in
%     pixels) of the spatial bins. For instance, if there are there
%     are 4 spatial bins along each spatial direction, the
%     ``side'' of the descriptor is approximatively 4 * MAGNIF.
%
%   WindowSize:: 2
%     Set the variance of the Gaussian window that determines the
%     descriptor support. It is expressend in units of spatial
%     bins.
%
%   Frames::
%     If specified, set the frames to use (bypass the detector). If
%     frames are not passed in order of increasing scale, they are
%     re-orderded.
%
%   Orientations::
%     If specified, compute the orientations of the frames overriding
%     the orientation specified by the 'Frames' option.
%
%   Verbose::
%     If specified, be verbose (may be repeated to increase the
%     verbosity level).
%
%   REFERENCES::
%     [1] D. G. Lowe, Distinctive image features from scale-invariant
%     keypoints. IJCV, vol. 2, no. 60, pp. 91-110, 2004.
%
%   See also: <a href="matlab:vl_help('sift')">SIFT</a>
%   VL_UBCMATCH(), VL_DSIFT(), VL_HELP().

% Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
    """

    nikeys = -1



    #/* -----------------------------------------------------------------
    #*                                               Check the arguments
    #* -------------------------------------------------------------- */

    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    if not isinstance(octaves, int) or octaves < 0:
        raise ValueError("'Octaves' must be a nonnegative integer.")

    if not isinstance(levels, int) or levels < 1:
        raise ValueError("'Levels' must be a positive integer.")

    if not isinstance(first_octave, int) :
        raise ValueError("'FirstOctave' must be an integer")
      #o_min = (int) *mxGetPr(optarg) ;

    if edge_thresh < 1:
        raise ValueError("'EdgeThresh' must be not smaller than 1.")

    if peak_thresh < 0:
        raise ValueError("'PeakThresh' must be a non-negative real.")

    if norm_thresh < 0:
        raise ValueError("'NormThresh' must be a non-negative real.")

    if magnif < 0:
        raise ValueError("'Magnif' must be a non-negative real.")

    if window_size < 0:
        raise ValueError("'WindowSize' must be a non-negative real.")

    if frames is not None:
        keys=np.asarray(frames)
        if keys.ndim<>2 or keys.shape[0]<>4:
           raise ValueError("'Frames' must be a 4 x N matrix.")
    else:
        keys=np.empty((0,0))

    image = np.require(image, dtype=np.float32, requirements='F')

    frames, descriptors = \
        cysift.sift(image, keys, octaves, levels, first_octave,
                    edge_thresh, peak_thresh, norm_thresh,
                    magnif, window_size, force_orientations,
                    verbose, return_descriptors, float_descriptors)
    return frames, np.ascontiguousarray(descriptors)
