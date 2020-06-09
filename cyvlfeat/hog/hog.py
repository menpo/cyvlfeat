import numpy as np
from .cyhog import cy_hog


def hog(image, cell_size, variant='UoCTTI', n_orientations=9,
        directed_polar_field=False, undirected_polar_field=False,
        bilinear_interpolation=False, verbose=False):
    r"""
    Computes the HOG features for ``image``
    and the specified ``cell_size``. ``image`` can be either greyscale or RGB
    and must be ``float32``.

    HOG decomposes an image into small squared cells, computes an histogram of
    oriented gradients in each cell, normalizes the result using a block-wise
    pattern, and return a descriptor for each cell.

    In this case the feature has 31 dimensions. HOG exists in many variants.
    VLFeat supports two: the UoCTTI variant (used by default) and the original
    Dalal-Triggs variant (with 2x2 square HOG blocks for normalization).
    The main difference is that the UoCTTI variant computes both directed and
    undirected gradients as well as a four dimensional texture-energy feature,
    but projects the result down to 31 dimensions.

    ``output`` is a ``float32`` array, its number
    of columns is approximately the number of columns of ``image`` divided
    by ``cell_size`` and the same for the number of rows. The third
    dimension spans the number of feature components.

    Parameters
    ----------
    image : [H, W] or [H, W, 1] or [H, W, 3] `float32` `ndarray`
        A single channel, RGB or greyscale, `float32` numpy array (ndarray)
        representing the image to calculate descriptors for.
    cell_size : `int`
        The size of the cells that the image should be decomposed into. Assumed
        to be square and this only a single integer is required.
    variant : `str`, optional
        Choose a HOG variant from the set {'UoCTTI', 'DalalTriggs'}
    n_orientations : `int`, optional
        Choose a number of undirected orientations in the orientation
        histograms. The angle ``[0, pi)`` is divided in to ``n_orientations``
        equal parts.
    directed_polar_field : `bool`, optional
        By specifying this flag, the image is interpreted as samples
        from a 2D vector field. Angles are measure clockwise, the y-axis
        pointing downwards, starting from the x-axis (pointing to the right).
        The first channel should be the gradient magnitude and the second
        channel should be the directed (``[0, 2pi)``) orientations.
    undirected_polar_field : `bool`, optional
        By specifying this flag, the image is interpreted as samples
        from a 2D vector field as above, however, the angles will be forcibly
        wrapped into the range ``[0, pi)``.
    bilinear_interpolation : `bool`, optional
        This flags activates the use of bilinear interpolation to assign
        orientations to bins. This produces a smoother feature, but is
        only used for the 'DalalTriggs' variant.
    verbose : `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    output : `(R, R, D)` `float32` `ndarray`
        ``R`` is is approximately the number of columns of ``image`` divided
        by ``cell_size``. ``D`` is the number of feature dimensions.
    """
    # Add a channels axis
    if image.ndim == 2:
        image = image[..., None]

    # Validate image size
    if image.ndim != 3:
        raise ValueError('Only 2D arrays with a channels axis are supported')

    # Validate all the parameters
    if cell_size < 0:
        raise ValueError('cell_size must be > 0')
    if variant not in {'UoCTTI', 'DalalTriggs'}:
        raise ValueError("variant must be in set {'UoCTTI', 'DalalTriggs'}")
    if n_orientations < 0:
        raise ValueError('n_orientations must be > 0')
    if (directed_polar_field or undirected_polar_field) and image.shape[-1] != 2:
        raise ValueError('Expected a polar field image of n_channels == 2')

    # Ensure types are correct before passing to Cython
    image = np.require(image, dtype=np.float32, requirements='C')

    # Shortcut for getting the correct enum value, since is_UoCTTI has
    # an enum value of 1 (True is 1 in Python)
    is_UoCTTI = variant == 'UoCTTI'

    return cy_hog(image, cell_size, is_UoCTTI,
                  n_orientations,  directed_polar_field,
                  undirected_polar_field, bilinear_interpolation,
                  True,  # Return channels as last axis
                  verbose)
