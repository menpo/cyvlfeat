import math
import numpy as np
from matplotlib import colors as colors
import scipy.ndimage
from . import dsift
from ..utils import conversion as conv


def plotsiftdescriptor(d, f, magnification_factor=3.0, num_spatial_bins=4, num_orientation_bins=8, max_value=0):
    r"""
    plotsiftdescriptor(D) plots the SIFT descriptor D. If D is a matrix,
    it plots one descriptor per column. D has the same format used by sift().
    plotsiftdescriptor(D,F) plots the SIFT descriptors warped to
    the SIFT frames F, specified as columns of the matrix F. F has the
    same format used by sift().
    H=plotsiftdescriptor(...) returns the handle H to the line
    drawing representing the descriptors.

    The function assumes that the SIFT descriptors use the standard
    configuration of 4x4 spatial bins and 8 orientations bins. The
    following parameters can be used to change this:

    Parameters
    ----------
    num_spatial_bins:: 4
      Number of spatial bins in both spatial directions X and Y.

    num_orientation_bins:: 8
      Number of orientation bis.

    magnification_factor:: 3
       Magnification factor. The width of one bin is equal to the scale
       of the keypoint F multiplied by this factor.
    """

    # Check the arguments
    if d:
        if not conv.is_numeric(f):
            #write test for non numeric types in plotsiftdescriptor
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
            # first five rows of f (python): five rows and all columns (matlab)
            f = np.row_stack((f, 10* np.ones((1, f.shape[1])), 0* np.zeros(1, f.shape[1])))

        if f.shape[0] == 3:
            # translation and scale
            m = np.array([1, 0, 0, 1])
            # might need a reshape here in place of transpose
            # do check the deep copy/no copy
            f = np.row_stack((f, (m.T)* f[2]))

        if f.shape[0] == 4:
            c = np.cos(f[3])
            s = np.sin(f[3])
            t = np.row_stack((c, s, -s, c))
            t = t.reshape(t.shape[0], 1)
            f = np.row_stack((f, f[3]*t))

        if f.shape[0] == 5:
            raise ValueError('Assertion Failed')

    # Standardizing Descriptors: Descriptors are often non-double numeric arrays
    d = d.astype(np.int32)
    K = d.shape[1]
