import numpy as np
from cyvlfeat.misc.cylbp import cy_lbp


def lbp(image, cell_size):
    r"""
    Computes the Local Binary Pattern (LBP) features for ``image``.
    ``image`` is divided in cells of size ``cell_size``.
    Parameters
    ----------
    image : [H, W] or [H, W, 1] `float32` `ndarray`
        A single channel, greyscale, `float32` numpy array (ndarray).
    cell_size : ``int``
        The ``image`` is divided in cells of size ``cell_size``.

    Returns
    -------
    histograms: `(h, w, 58)` `float32` `ndarray`
        `h = FLOOR(height/cell_size)`
        `w FLOOR(width/cell_size)`
        ``histograms`` is a three-dimensional array containing one histograms of quantized
        LBP features per cell. The width of ``histograms`` is ``FLOOR(width/cell_size)``,
        where ``width`` is the width of the image. The same for the ``height``.
        The third dimension is 58.
    """

    # check for none
    if image is None or cell_size is None:
        raise ValueError('One of the required arguments is None')

    # Remove last channel
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    if cell_size < 1:
        raise ValueError('cell_size is less than 1')

    # Ensure types are correct before passing to Cython
    image = np.require(image, dtype=np.float32, requirements='C')

    histograms = cy_lbp(image, cell_size)

    return histograms
