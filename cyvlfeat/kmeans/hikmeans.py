from __future__ import division

import numpy as np
from .cykmeans import cy_hikmeans, cy_hikmeans_push, algorithm_type_ikmeans
import math


def hikmeans(data, num_centers, num_leaves, algorithm='LLOYD', verbose=False):
    """
    Hierachical integer K-means

    Parameters
    ----------
    data : [N, D] `uint8` `ndarray`
        Data to be clustered
    num_centers : `int`
        Number of clusters (leaves) per level
    num_leaves : `int`
        Minimum number of final leaves in the tree. The depth of the tree
        is computed from this parameter and ``num_centers`
    algorithm : {'LLOYD', 'ELKAN'}, optional
        Algorithm to be used for clustering.
    verbose : `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    (tree_structure, assignments) : (object, [N, depth] `uint32` `ndarray`)
        Computed organisation and per-level assignments for ``data``
    """
    assert isinstance(data, np.ndarray)
    assert isinstance(num_centers, int)
    assert isinstance(num_leaves, int)
    assert isinstance(verbose, bool)
    if data.ndim != 2:
        raise ValueError('Data should be a 2-D matrix')

    if data.dtype != np.uint8:
        raise ValueError('Data should be uint8')

    if num_centers > data.shape[0]:
        raise ValueError('num_centers should be a positive integer smaller '
                         'than the number of data points')

    algorithm_b = algorithm.encode()
    if algorithm_b not in algorithm_type_ikmeans.keys():
        raise ValueError('algorithm field invalid')

    depth = max(1, math.ceil(math.log(num_leaves) / math.log(num_centers)))

    return cy_hikmeans(data, num_centers, num_leaves, depth, algorithm_b,
                       verbose)


def hikmeans_push(data, py_tree, verbose=False):
    """
    Projects the data on the HKMeans nearest elements.

    Parameters
    ----------
    data : [N, D] `uint8` `ndarray`
        Data to be projected
    py_tree : `PyHIKMNode`
        Tree structure created by ``hikmeans``
    verbose : bool, optional
        If ``True``, be verbose.

    Returns
    -------
    assignments : [N, depth] `uint32` `ndarray`
        Computed per-level assignments for ``data``
    """

    return cy_hikmeans_push(data, py_tree, verbose)
