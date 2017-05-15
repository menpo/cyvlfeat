import numpy as np
from cyvlfeat.sift.cysift import cy_siftdescriptor


def siftdescriptor(gradient_image, frames, magnification=3,
                   float_descriptors=False, norm_thresh=None, verbose=False):
    r"""
    Calculates the SIFT descriptors of the keypoints ``frames`` on the
     pre-processed image ``gradient_image``.

    In order to match the standard SIFT descriptor, the gradient should be
    calculated after mapping the image to the keypoint scale. This is obtained
    by smoothing the image by a a Gaussian kernel of variance equal to the scale
    of the keypoint. Additionally, SIFT assumes that the input image is
    pre-smoothed at scale 0.5 (this roughly compensates for the effect of the
    CCD integrators), so the amount of smoothing that needs to be applied is
    slightly less.

    Parameters
    ----------
    grad : [2, H, W] `float32` `ndarray`
        gradient_image is a 2xMxN array. The first channel
        ``gradient_image[0, :, :]`` contains the
        modulus of gradient of the original image modulus. The second channel
        `gradient_image[1, :, :]`` contains the gradient angle (measured in
        radians, clockwise, starting from the X axis -- this assumes that the Y
        axis points down).
    frames : `[F, 4]` `float32` `ndarray`, optional
        If specified, set the frames to use (bypass the detector). If frames are
        not passed in order of increasing scale, they are re-orderded. A frame
        is a vector of length 4 ``[Y, X, S, TH]``, representing a disk of center
        frames[:2], scale frames[2] and orientation frames[3].
    magnification : `int`, optional
        Set the descriptor magnification factor. The scale of the keypoint is
        multiplied by this factor to obtain the width (in pixels) of the spatial
        bins. For instance, if there are there are 4 spatial bins along each
        spatial direction, the ``side`` of the descriptor is approximately ``4 *
        magnification``.
    float_descriptors : `bool`, optional
        If ``True``, the descriptor are returned in floating point rather than
        integer format.
    norm_thresh : `float`, optional
        Set the minimum l2-norm of the descriptors before normalization.
        Descriptors below the threshold are set to zero. If ``None``,
        norm_thresh is ``-inf``.
    verbose : `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    descriptors : `(F, 128)` `uint8` or `float32` `ndarray`, optional
        ``F`` is the number of keypoints (frames) used. The 128 length vectors
        per keypoint extracted. ``uint8`` by default.


    Examples
    --------
    >>> import scipy.ndimage
    >>> import numpy as np
    >>> from cyvlfeat.sift import siftdescriptor
    >>> from cyvlfeat.test_util import lena
    >>> img = lena().astype(np.float32)
    >>> # Create a single frame in the center of the image
    >>> frames = np.array([[256, 256, 1.0, np.pi / 2]])
    >>> sigma = np.sqrt(frames[0, 2] ** 2 - 0.25)  # 0.25 = 0.5^2
    >>> img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
    >>> y, x = np.gradient(img_smooth)
    >>> mod = np.sqrt(x * x + y * y)
    >>> ang = np.arctan2(y, x)
    >>> gradient_image = np.stack((mod, ang), axis=0)
    >>>
    >>> d = siftdescriptor(gradient_image, frames, verbose=True)

    Notes
    -----
    1. The above fragment generates results which are very close
       but not identical to the output of ``sift`` as the latter
       samples the scale space at finite steps.

    2. For object categorization is sometimes useful to compute
       SIFT descriptors without smoothing the image.
    """

    # Validate Gradient array size
    if gradient_image.ndim != 3:
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
    gradient_image = np.require(gradient_image, dtype=np.float32,
                                requirements='C')
    frames = np.require(frames, dtype=np.float32, requirements='C')

    return cy_siftdescriptor(gradient_image, frames, magnification,
                             float_descriptors, norm_thresh, verbose)
