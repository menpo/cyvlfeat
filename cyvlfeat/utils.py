import numpy as np


def rgb2gray(rgb):
    r"""Convert RGB image to Greyscale.

    Parameters
    ----------
    rgb : Numpy array for RGB Image with shape[2] = 3
          The array to convert.

    Returns
    -------
    Numpy array for Greyscale Image.

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])