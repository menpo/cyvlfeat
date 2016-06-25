import numpy as np


def rgb2gray(rgb):
    r"""Convert RGB image to Grayscale.

    Parameters
    ----------
    rgb : Numpy array for RGB Image with shape[2] = 3
          The array to convert.

    Returns
    -------
    Numpy array for Grayscaled Image.

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set('buifc')


def is_numeric(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS

