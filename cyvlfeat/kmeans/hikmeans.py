from __future__ import division

import numpy as np
from .cykmeans import cy_hikmeans, cy_hikmeans_push, algorithm_type_ikmeans
import math


def hikmeans(data, num_centers, num_leaves, algorithm='LLOYD', verbose=False):
    """
    Hierachical integer K-means

    Parameters
    ----------
    data : [N,D] np.ndarray of type np.uint8
        Data to be clustered
    num_centers : int
        Number of clusters (leaves) per level
    num_leaves : int
        Number of final leaves in the tree
    algorithm : str, optional
        Algorithm to be used for clustering ('LLOYD', 'ELKAN')
    verbose : bool, optional

    Returns
    -------
    (tree_structure, assignments) : (<PyObject>, [N,depth] np.ndarray of type np.uint32)
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
        raise ValueError('num_centers should be a positive integer smaller than the number of data points')

    algorithm_b = algorithm.encode()
    if algorithm_b not in algorithm_type_ikmeans.keys():
        raise ValueError('algorithm field invalid')

    depth = max(1, math.ceil(math.log(num_leaves) / math.log(num_centers)))

    return cy_hikmeans(data, num_centers, num_leaves, depth, algorithm_b, verbose)


def hikmeans_push(data, py_tree, verbose=False):
    """
    Projects the data on the HKMeans nearest elements
    Parameters
    ----------
    data : [N,D] np.ndarray of type np.uint8
        Data to be projected
    py_tree :
        Tree structure created by ``hikmeans``
    verbose : bool, optional

    Returns
    -------
    assignments : [N,depth] np.ndarray of type np.uint32
        Computed per-level assignments for ``data``
    """

    return cy_hikmeans_push(data, py_tree, verbose)
