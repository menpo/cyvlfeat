import numpy as np
from .cysift import cy_dsift


def dsift(image, step=1, size=3, bounds=None, window_size=-1, norm=False,
          fast=False, float_descriptors=False, geometry=(4, 4, 8),
          verbose=False):
    r"""
    Extracts a dense set of SIFT features from ``image``. ``image`` must be
    ``float32`` and greyscale (either a single channel as the last axis, or no
    channel).

    **Important!:** Unlike the Matlab wrapper of vlfeat, this method always
    takes the input arguments as Y-axis, then X-axis. For example, the step
    [2, 3] denotes 2 pixels in the Y-axis, then two pixels in the X-axis. This
    is to create a 1-1 mapping from the input parameters to the shape of the
    image, which is always denoted [H, W] (size of Y-domain, size of X-domain).
    Also, note that the frames are not 1-based as in Matlab, and therefore
    the top left corner is ``(0, 0)`` not ``(1, 1)``.

    Does NOT compute a Gaussian scale space of the image. Instead, the image
    should be pre-smoothed at the desired scale level.

    The scale of the extracted descriptors is controlled by the option
    ``size``, i.e. the width in pixels of a spatial bin (recall that a
    SIFT descriptor is a spatial histogram with 4 x 4 bins).

    The sampling density is controlled by the option ``step``, which is
    the horizontal and vertical displacement of each feature center to
    the next.

    The sampled image area is controlled by the option ``bounds``,
    defining a rectangle in which features are computed. A descriptor
    is included in the rectangle if all the centers of the spatial
    bins are included. The upper-left descriptor is placed so that the
    upper-left spatial bin center is aligned with the upper-left
    corner of the rectangle.

    By default, ``dsift`` computes features equivalent to
    ``sift``. However, the ``fast`` option can be used to turn on an
    variant of the descriptor which, while not strictly equivalent, is much
    faster.

    **Relation to SIFT**

    In the standard SIFT detector/descriptor, implemented by
    ``sift``, the size of a spatial bin is related to the keypoint
    scale by a multiplier, called magnification factor, and denoted
    `magnification`. Therefore, the keypoint scale corresponding to the
    descriptors extracted by ``dsift`` is equal to ``size`` /
    ``magnification``. ``dsift`` does not use ``magnification`` because, by
    using dense sampling, it avoids detecting keypoints in the first place.

    ``dsift`` does not smooth the image as SIFT does. Therefore, in
    order to obtain equivalent results, the image should be
    pre-smoothed appropriately. Recall that in SIFT, for a keypoint of
    scale ``S``, the image is pre-smoothed by a Gaussian of variance
    ``S.^2 - 1/4``.

    **Further details on the geometry**

    As mentioned, the ``dsift`` descriptors cover the bounding box
    specified by ``bounds = [YMIN, XMIN, YMAX, XMAX]``. Thus the top-left bin
    of the top-left descriptor is placed at ``(YMIN, XMIN)``. The next
    three bins to the right are at ``YMIN + size``, ``YMIN + 2*size``,
    ``YMIN + 3*size``. The Y coordinate of the center of the first descriptor is
    therefore at ``(YMIN + YMIN + 3*size) / 2 = YMIN + 3/2 * size``. For
    instance, if ``YMIN=1`` and ``size=3`` (default values), the Y
    coordinate of the center of the first descriptor is at
    ``1 + 3/2 * 3 = 5.5``. For the second descriptor immediately to its right
    this is ``5.5 + step``, and so on.

    Parameters
    ----------
    image : [H, W] or [H, W, 1] `float32` `ndarray`
        A single channel, greyscale, `float32` numpy array (ndarray)
        representing the image to calculate descriptors for.
    step : `int`, optional
        A SIFT descriptor is extracted every ``step`` pixels. This allows for
        sub-sampling of the image.
    size : `int`, optional
        The size of the spatial bin of the SIFT descriptor in pixels.
    bounds : [`int`, `int`, `int`, `int`], optional
        Specifies a rectangular area where descriptors should be
        extracted. The format is ``[YMIN, XMIN, YMAX, XMAX]``. If this
        option is not specified, the entire image is used. The
        bounding box is clipped to the image boundaries.
    norm : `bool`, optional
        If ``True``, adds to the ``frames`` output argument a third
        row containing the descriptor norm, or energy, before
        contrast normalization. This information can be used to
        suppress low contrast descriptors.
    fast : `bool`, optional
        If ``True``, use a piecewise-flat, rather than Gaussian,
        windowing function. While this breaks exact SIFT equivalence,
        in practice is much faster to compute.
    float_descriptors : `bool`, optional
        If ``True``, the descriptor are returned in floating point
        rather than integer format.
    geometry : [`int`, `int`, `int`], optional
        Specify the geometry of the descriptor as ``[NY, NX, NO]``, where ``NY``
        is the number of bins in the Y direction, NX in the X direction,
        and NO the number of orientation bins.
    verbose : `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    frames : `(F, 2)` or `(F, 3)` `float32` `ndarray`
        ``F`` is the number of keypoints (frames) used. This is the center
        of every dense SIFT descriptor that is extracted.
    descriptors : `(F, 128)` `uint8` or `float32` `ndarray`
        ``F`` is the number of keypoints (frames) used. The 128 length vectors
        per keypoint extracted. ``uint8`` by default.
    """
    # Remove last channel
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    # Validate bounds
    if bounds is None:
        bounds = np.array([0, 0, image.shape[0] - 1, image.shape[1] - 1])
    else:
        bounds = np.asarray(bounds)
    if bounds.shape[0] != 4:
        raise ValueError('Bounds must be contain 4 elements.')
    for b in bounds:
        if b < 0:
            raise ValueError('Bounds must only contain integers greater than '
                             'or equal to 0.')
    if bounds[2] > image.shape[0] - 1 or bounds[3] > image.shape[1] - 1:
        raise ValueError('Bounds must be the size of the image or less.')

    # Validate size
    if isinstance(size, int):
        size = np.array([size, size])
    else:
        size = np.asarray(size)
    if size.shape[0] != 2:
        raise ValueError('Size vector must contain exactly 2 elements.')
    for s in size:
        if s < 1:
            raise ValueError('Size must only contain positive integers.')

    # Validate step
    if isinstance(step, int):
        step = np.array([step, step])
    else:
        step = np.asarray(step)
    if step.shape[0] != 2:
        raise ValueError('Step vector must contain exactly 2 elements.')
    for s in step:
        if s < 1:
            raise ValueError('Step must only contain positive integers.')

    # Validate window_size
    if not isinstance(window_size, int):
        raise ValueError('Window size must be an integer.')

    # Validate geometry
    geometry = np.asarray(geometry)
    if geometry.shape[0] != 3:
        raise ValueError('Geometry must contain exactly 3 integer elements.')
    if np.min(geometry) < 1:
        raise ValueError('Geometry must only contain positive integers.')

    geometry = geometry.astype(np.int32)
    step = step.astype(np.int32)
    size = size.astype(np.int32)
    bounds = bounds.astype(np.int32)
    image = np.require(image, dtype=np.float32, requirements='C')
    frames, descriptors = cy_dsift(image, step, size, bounds, window_size,
                                   norm, fast, float_descriptors, geometry,
                                   verbose)
    return frames, descriptors
