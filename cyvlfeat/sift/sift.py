import numpy as np
from .cysift import cy_sift


def sift(image, n_octaves=None, n_levels=3,  first_octave=0,  peak_thresh=0,
         edge_thresh=10, norm_thresh=None,  magnification=3, window_size=2,
         frames=None, force_orientations=False, float_descriptors=False,
         compute_descriptor=False, verbose=False):
    r"""
    Extracts a set of SIFT features from ``image``. ``image`` must be
    ``float32`` and greyscale (either a single channel as the last axis, or no
    channel). Each column of ``frames`` is a feature frame and has the format
    ``[Y, X, S, TH]``, where ``(Y, X)`` is the floating point center of the
    keypoint, ``S`` is the scale and ``TH`` is the orientation (in radians).

    If ``compute_descriptors=True``, computes the SIFT descriptors as well. Each
    column of ``descriptors`` is the descriptor of the corresponding frame in
    ``frames``. A descriptor is a 128-dimensional vector of type ``uint8``.

    Parameters
    ----------
    image : [H, W] or [H, W, 1] `float32` `ndarray`
        A single channel, greyscale, `float32` numpy array (ndarray)
        representing the image to calculate descriptors for.
    n_octaves : `int`, optional
        The number of octaves of the DoG scale space. If ``None``, the maximum
        number of octaves will be used.
    n_levels : `int`, optional
        The number of levels per octave of the DoG scale space.
    first_octave : `int`, optional
        The index of the first octave of the DoG scale space.
    peak_thresh : `float`, optional
        The peak selection threshold. The peak threshold filters peaks of the
        DoG scale space that are too small (in absolute value).
    edge_thresh : `float`, optional
        The edge selection threshold. The edge threshold eliminates peaks of the
        DoG scale space whose curvature is too small (such peaks yield badly
        localized frames).
    norm_thresh : `float`, optional
        Set the minimum l2-norm of the descriptors before normalization.
        Descriptors below the threshold are set to zero. If ``None``,
        norm_thresh is ``-inf``.
    magnification : `int`, optional
        Set the descriptor magnification factor. The scale of the keypoint is
        multiplied by this factor to obtain the width (in pixels) of the spatial
        bins. For instance, if there are there are 4 spatial bins along each
        spatial direction, the ``side`` of the descriptor is approximately ``4 *
        magnification``.
    window_size : `int`, optional
        Set the variance of the Gaussian window that determines the
        descriptor support. It is expressed in units of spatial bins.
    frames : `[F, 4]` `float32` `ndarray`, optional
        If specified, set the frames to use (bypass the detector). If frames are
        not passed in order of increasing scale, they are re-orderded. A frame
        is a vector of length 4 ``[Y, X, S, TH]``, representing a disk of center
        f[:2], scale f[2] and orientation f[3].
    force_orientations : `bool`, optional
        If ``True``, compute the orientations of the frames, overriding the
        orientation specified by the ``frames`` argument.
    float_descriptors : `bool`, optional
        If ``True``, the descriptor are returned in floating point rather than
        integer format.
    compute_descriptor : `bool`, optional
        If ``True``, the descriptors are also returned, as well as the keypoints
        (frames). This means that the output of calling this function changes
        from a single value ``frames``, to a tuple of output values ``(frames,
        descriptors)``.
    verbose : `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    frames : `(F, 4)` `float32` `ndarray`
        ``F`` is the number of keypoints (frames) used. This is the center
        of every dense SIFT descriptor that is extracted.
    descriptors : `(F, 128)` `uint8` or `float32` `ndarray`, optional
        ``F`` is the number of keypoints (frames) used. The 128 length vectors
        per keypoint extracted. ``uint8`` by default. Only returned if
        ``compute_descriptors=True``.
    """
    # Remove last channel
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    if frames is not None:
        if frames.ndim != 2 or frames.shape[-1] != 4:
            raise ValueError('Frames should be a 2D array of size '
                             '(n_keypoints, 4)')
        frames = np.require(frames, dtype=np.float32, requirements='C')

    # Validate all the parameters
    if n_octaves is not None and n_octaves < 0:
        raise ValueError('n_octaves must be >= 0')
    if n_octaves is None:
        n_octaves = -1
    if n_levels < 1:
        raise ValueError('n_levels must be > 0')
    if first_octave < 0:
        raise ValueError('first_octave must be >= 0')
    if edge_thresh <= 0:
        raise ValueError('edge_thresh must be > 0')
    if peak_thresh < 0:
        raise ValueError('peak_thresh must be >= 0')
    if norm_thresh is not None and norm_thresh < 0:
        raise ValueError('norm_thresh must be >= 0')
    if norm_thresh is None:
        norm_thresh = -1
    if window_size < 0:
        raise ValueError('window_size must be >= 0')

    # Ensure types are correct before passing to Cython
    image = np.require(image, dtype=np.float32, requirements='C')

    result = cy_sift(image, n_octaves, n_levels,
                     first_octave,  peak_thresh,
                     edge_thresh, norm_thresh,  magnification,
                     window_size, frames, force_orientations,
                     float_descriptors, compute_descriptor,
                     verbose)
    # May be a tuple or a single return of only the calculated frames
    return result
