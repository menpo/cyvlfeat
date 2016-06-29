import math
import numpy as np
from numpy import newaxis
from numpy import matlib
from matplotlib import collections as mc
from matplotlib import pyplot as plt
from cyvlfeat.utils import utils as utils
from cyvlfeat.sift.sift import sift


def plotsiftdescriptor(d, f=None, magnification=3.0, num_spatial_bins=4, num_orientation_bins=8, max_value=0):
    r"""
    Plots the SIFT descriptor ``d``. If ``d`` is a
    matrix, it plots one descriptor per column. ``d`` has the same format
    used by ``sift()``.

    ``plotsiftdescriptor(d,f)`` plots the SIFT descriptors warped to
    the SIFT frames ``f``, specified as columns of the matrix ``f``. ``f`` has the same format used by ``sift()``.

    The function assumes that the SIFT descriptors use the standard
    configuration of 4x4 spatial bins and 8 orientations bins. The
    parameters ``num_spatial_bins`` and ``num_orientation_bins`` respectively can be used to change this:

    Parameters
    ----------
    d : `(F, 128)` `uint8` or `float32` `ndarray`
        ``descriptors as returned by ``sift()``
    f : `float32` `ndarray`, optional
        ``frames`` as returned by ``sift()``
        By default
        >>> f = np.matlib.repmat(np.array([[0], [0], [1], [0], [0], [1]]), 1, d.shape[0])
    magnification :  `int`, optional
        Set the descriptor magnification factor. The scale of the keypoint is
        multiplied by this factor to obtain the width (in pixels) of the spatial
        bins. For instance, if there are there are 4 spatial bins along each
        spatial direction, the ``side`` of the descriptor is approximately ``4 *
        magnification``.
    num_spatial_bins : `int`, optional
        Number of spatial bins in both spatial directions X and Y.
    num_orientation_bins : `int`, optional
        Number of orientation bis.
    max_value : `int`, optional

    Example Usage
    -------------
    >>> result = sift(img, compute_descriptor=True)
    >>> F = result[0]
    >>> D = result[1]
    >>> plotsiftdescriptor(D, F)
    >>> #or
    >>> plotsiftdescriptor(D) # Default f will be used in this case.
    """

    # Check the arguments
    if d:
        if not conv.is_numeric(f):
            # write test for non numeric types in plotsiftdescriptor
            raise ValueError('F must be a numeric type')

        if d.shape[0] != (math.pow(num_spatial_bins, 2) * num_orientation_bins):
            raise ValueError('The number of rows of D does not match the geometry of the descriptor')
    else:
        raise ValueError('Not enough arguments')

    if f:
        if f.shape[1] != d.shape[1]:
            raise ValueError('D and F have incompatible dimension')

        if f.shape[0] < 2 or f.shape[0] > 6:
            raise ValueError('F must be either empty or have from 2 to six rows.')

        if f.shape[0] == 2:
            # translation only
            f = np.row_stack((f, 10 * np.ones((1, f.shape[1])), 0 * np.zeros(1, f.shape[1])))

        if f.shape[0] == 3:
            # translation and scale
            m = np.array([1, 0, 0, 1])
            # check deep copy/no copy
            f = np.row_stack((f, m * f[2]))

        if f.shape[0] == 4:
            c = np.cos(f[3])
            s = np.sin(f[3])
            t = np.row_stack((c, s, -s, c))
            f = np.row_stack((f, f[3] * t))
    else:
        f = np.matlib.repmat(np.array([[0], [0], [1], [0], [0], [1]]), 1, d.shape[1])

    # Standardizing Descriptors: Descriptors are often non-double numeric arrays
    d = d.astype(np.int32)

    def render_descr(d, num_spatial_bins, num_orientation_bins, max_value):
        r"""
        render_descr(d, num_spatial_bins, num_orientation_bins, max_value)

        Parameters
        ----------
        num_spatial_bins:: 4
          Number of spatial bins in both spatial directions X and Y.

        num_orientation_bins:: 8
          Number of orientation bis.
        """

        # Get the coordinates of the lines of the SIFT grid; each bin has side 1
        x, y = np.meshgrid(np.arange(-num_spatial_bins / 2, num_spatial_bins / 2 + 1),
                           np.arange(-num_spatial_bins / 2, num_spatial_bins / 2 + 1))

        # Get the corresponding bin centers
        # a copy will be made and that'll be assigned to new vars.
        xc = x[:, :-1] + 0.5
        yc = y[:, :-1] + 0.5

        # Rescale the descriptor range so that the biggest peak fits inside the bin diagram
        d = 0.4 * d / max_value

        # Each spatial bin contains a star with numOrientationBins tips
        # flatten() Scramble the the centers automatically in row major order(descriptor convention)
        xc = np.matlib.repmat(xc.flatten(), num_orientation_bins, 1)
        yc = np.matlib.repmat(yc.flatten(), num_orientation_bins, 1)

        # Do the stars
        th = np.linspace(0, 2 * math.pi, num_orientation_bins + 2)
        th = th[1:]
        xd = np.matlib.repmat(np.cos(th), 1, num_spatial_bins * num_spatial_bins)
        yd = np.matlib.repmat(np.sin(th), 1, num_spatial_bins * num_spatial_bins)
        # d in Matlab is a 128x1 double but here it is 128x1 array, hence no transposing required.
        xd = xd * d.flatten()
        yd = yd * d.flatten()

        # Re-arrange in sequential order the lines to draw
        nans = np.empty(num_spatial_bins * num_spatial_bins * num_orientation_bins)
        nans[:] = np.NAN
        x1 = xc.flatten()
        y1 = yc.flatten()
        x2 = x1 + xd
        y2 = y1 + yd
        xstars = np.row_stack((x1, x2, nans))
        ystars = np.row_stack((y1, y2, nans))

        # Horizontal lines of the grid
        nans = np.empty(num_spatial_bins + 1)
        xh = np.row_stack((x[:, 1][:, newaxis], x[:, -1][:, newaxis], nans[:, newaxis]))
        yh = np.row_stack((y[:, 1][:, newaxis], y[:, -1][:, newaxis], nans[:, newaxis]))

        # Vertical lines of the grid
        # check for copy
        xv = np.row_stack((x[1, :], x[-1, :], nans))
        yv = np.row_stack((y[1, :], y[-1, :], nans))

        x = np.hstack((xstars.flatten, xh.flatten, xv.flatten))
        y = np.hstack((ystars.flatten, yh.flatten, yv.flatten))

        return x, y

    d_len = d.shape[1]
    xall = []
    yall = []

    for k in range(d_len):
        x, y = render_descr(d[:, k], num_spatial_bins, num_orientation_bins, max_value)
        xall = np.hstack((xall, magnification_factor * f[2, k] * x + magnification_factor * f[4, k] * y + f[0, k]))
        yall = np.hstack((yall, magnification_factor * f[3, k] * x + magnification_factor * f[5, k] * y + f[1, k]))

    # Plotting
    plot_figure = plt.figure()
    to_plot = [list(zip(xall, yall))]
    lc = mc.LineCollection(to_plot, linewidths=2)
    plt.gca().add_collection(lc)
    plt.autoscale(enable=True, axis='both', tight=None)
