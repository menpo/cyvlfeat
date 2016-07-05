import numpy as np
from cyvlfeat.sift.cysift import cy_siftdescriptor


def siftdescriptor(grad, f, magnification=3, float_descriptors=False, norm_thresh=None, verbose=False):
    r"""
    Calculates the SIFT descriptors of
    the keypoints F on the pre-processed image GRAD.

    In order to match the standard SIFT descriptor, the gradient GRAD
    should be calculated after mapping the image to the keypoint
    scale. This is obtained by smoothing the image by a a Gaussian
    kernel of variance equal to the scale of the keypoint.
    Additionally, SIFT assumes that the input image is pre-smoothed at
    scale 0.5 (this roughly compensates for the effect of the CCD
    integrators), so the amount of smoothing that needs to be applied
    is slightly less. The following code computes a standard SIFT
    descriptor by using SIFTDESCRIPTOR():

    Parameters
    ----------
    grad : `float32` `ndarray`
        grad is a 2xMxN array. The first layer GRAD[0,:,:] contains the
        modulus of gradient of the original image modulus. The second layer
        GRAD[1,:,:] contains the gradient angle (measured in radians,
        clockwise, starting from the X axis -- this assumes that the Y
        axis points down).
    f : `float32` `ndarray`
        `f` contains one row per keypoint with the `x`,`y`, `SIGMA` and `ANGLE` parameters.
    magnification : `int`, optional
        Set the descriptor magnification factor. The scale of the keypoint is
        multiplied by this factor to obtain the width (in pixels) of the spatial
        bins. For instance, if there are there are 4 spatial bins along each
        spatial direction, the ``side`` of the descriptor is approximately ``4 *
        magnification``.
    float_descriptors : `bool`, optional
        If ``True``, the descriptor are returned in floating point rather than
        integer format.
    verbose : `bool`, optional
        If ``True``, be verbose.
    norm_thresh : `float`, optional
        Set the minimum l2-norm of the descriptors before normalization.
        Descriptors below the threshold are set to zero. If ``None``,
        norm_thresh is ``-inf``.

    Returns
    -------
    descriptors : `(F, 128)` `uint8` or `float32` `ndarray`, optional
        ``F`` is the number of keypoints (frames) used. The 128 length vectors
        per keypoint extracted. ``uint8`` by default.

    The following code computes a standard SIFT
    descriptor by using `siftdescriptor()`:


    Examples
    --------
    >>> import scipy.ndimage
    >>> import math
    >>> import numpy as np
    >>> from cyvlfeat.test_util import lena
    >>> img = lena().astype(np.float32)
    >>> result = sift(img,compute_descriptor=True)
    >>> f = result[0]
    >>> sigma = math.sqrt(math.pow(f[3][2], 2) - math.pow(0.5, 2))
    >>> img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)  # smoothing
    >>> x, y = np.gradient(img_smooth)
    >>> mod = math.sqrt(math.pow(x, 2)+ math.pow(y, 2))
    >>> ang = np.arctan2(y, x) * 180 / np.pi
    >>> grad = np.rollaxis(np.dstack((mod, ang)), 1) # Stack arrays in sequence depth wise (along third axis).
    >>> d = siftdescriptor(grad, f)

    Notes
    -----
    1. The above fragment generates results which are very close
    but not identical to the output of VL_SIFT() as the latter
    samples the scale space at finite steps.

    2. For object categorization is sometimes useful to compute
    SIFT descriptors without smoothing the image.
    """

    # Validate Gradient array size
    if grad.ndim != 3:
        raise ValueError('Only 3D arrays are supported')

    # Validate magnification
    if magnification < 0:
        raise ValueError('Magnification must be non-negative')

    # Validate norm_thresh
    if norm_thresh is not None and norm_thresh < 0:
        raise ValueError('norm_thresh must be >= 0')
    if norm_thresh is None:
        norm_thresh = -1

    # Ensure types are correct before passing to Cython
    grad = np.require(grad, dtype=np.float32, requirements='C')
    f = np.require(f, dtype=np.float32, requirements='C')

    descriptors = cy_siftdescriptor(grad, f, magnification, float_descriptors, norm_thresh, verbose)

    return descriptors
