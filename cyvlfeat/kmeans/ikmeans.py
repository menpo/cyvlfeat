import numpy as np
from .cykmeans import cy_ikmeans, cy_ikmeans_push, algorithm_type_ikmeans


def ikmeans(data, num_centers, algorithm="LLOYD", max_num_iterations=200,
            verbose=False):
    """
    Integer K-means

    Parameters
    ----------
    data : [N, D] `uint8` `ndarray`
        Data to be clustered
    num_centers : `int`
        Number of clusters (leaves) per level
    algorithm : {'LLOYD', 'ELKAN'}, optional
        Algorithm to be used for clustering.
    max_num_iterations : `int`, optional
        Maximum number of iterations before giving up (the algorithm
        stops as soon as there is no change in the data to cluster
        associations).
    verbose : bool, optional
        If ``True``, be verbose.

    Returns
    -------
    (centers, assignments) : ([num_centers, D] `int32` `ndarray`, [N,] `uint32` `ndarray`)
        Computed centers of the clusters and their assignments
    """

    assert isinstance(data, np.ndarray)
    assert isinstance(num_centers, int)
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

    if (not isinstance(max_num_iterations, int)) or max_num_iterations <= 0:
        raise ValueError('max_num_iterations should be a positive integer')

    return cy_ikmeans(data, num_centers, algorithm_b, max_num_iterations,
                      verbose)


def ikmeans_push(data, centers):
    """
    Projects the data on the KMeans nearest elements (similar to
    kmeans_quantize but for integer data).

    Parameters
    ----------
    data : [N, D] `uint8` `ndarray`
        Data to be projected to the centers assignments
    centers : [K, D] `int32` `ndarray`
        Centers positions

    Returns
    -------
    assignments : [N,] `uint32` `ndarray`
        Assignments of the data points to their respective clusters indice.
    """
    assert isinstance(data, np.ndarray)

    if data.ndim != 2:
        raise ValueError('Data should be a 2-D matrix')
    if data.dtype != np.uint8:
        raise ValueError('Data should be uint8')
    if centers.ndim != 2:
        raise ValueError('Centers should be a 2-D matrix')
    if centers.dtype != np.int32:
        raise ValueError('Centers should be int32')

    return cy_ikmeans_push(data, centers)
