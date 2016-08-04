import numpy as np
from .cykmeans import cy_kmeans, cy_kmeans_quantize, algorithm_type, initialization_type, distance_type


def kmeans(data, num_centers, distance="l2", initialization="RANDSEL",
           algorithm="LLOYD", num_repetitions=1, num_trees=3,
           max_num_comparisons=100, max_num_iterations=100,
           min_energy_variation=0., verbose=False):
    """
    Clusters ``data`` using the K-means algorithm where ``data`` can be either
    `float32` or `float64`.
    Returns the centers of the clusters as an `ndarray` of the same dtype
    as ``data``. Also returns the cluster assignments.

    Parameters
    ----------
    data : [N, D] `float32/float64` `ndarray`
        Input data to be clustered.
    num_centers : `int`
        Number of clusters to compute
    distance : `str`, optional
        Distance to be used ["l1", "l2"]. Default : "l2"
    initialization : `str`, optional
        Use either random data points "RANDSEL" or k-means++ "PLUSPLUS"
        to initialize the centers. Default : "RANDSEL"
    algorithm : `str`, optional
        One of "LLOYD", "ELKAN", or "ANN". LLOYD is the standard Lloyd
        algorithm (similar to expectation maximisation). ELKAN is a
        faster version of LLOYD using triangular inequalities to cut
        down significantly the number of sample-to-center
        comparisons. ANN is the same as Lloyd, but uses an approximated
        nearest neighbours (ANN) algorithm to accelerate the
        sample-to-center comparisons. The latter is particularly
        suitable for very large problems. Default : "LLOYD"
    num_repetitions : `int`, optional
        Number of time to restart k-means. The solution with minimal
        energy is returned. Default : 1
    num_trees : `int`, optional
        The number of trees int the randomized KD-Tree forest (for "ANN").
    max_num_comparisons : `int`, optional
        Maximum number of sample-to-center comparisons when searching
        for the closest center (for "ANN").
    max_num_iterations : `int`, optional
        Maximum number of iterations allowed for the kmeans algorithm
        to converge (for "ANN").
    min_energy_variation : `float`, optional
        The minimum relative energy variation before the algorithm choose to
        stop.
    verbose : `bool`, optional
        Outputs information of the converging process

    Returns
    -------
    centers : [num_centers, D] `float32/float64` `ndarray`
        Computed clusters centers from the data points.
    """

    assert isinstance(num_centers, int)
    assert isinstance(verbose, bool)
    if data.ndim != 2:
        raise ValueError('Data should be a 2-D matrix')

    if data.dtype not in [np.float32, np.float64]:
        raise ValueError('Data should be float32 or float64')

    if num_centers > data.shape[0]:
        raise ValueError('num_centers should be a positive integer smaller '
                         'than the number of data points')

    distance_b = distance.encode()
    if distance_b not in distance_type.keys():
        raise ValueError('distance field invalid')

    initialization_b = initialization.encode()
    if initialization_b not in initialization_type.keys():
        raise ValueError('initialization field invalid')

    algorithm_b = algorithm.encode()
    if algorithm_b not in algorithm_type.keys():
        raise ValueError('algorithm field invalid')

    if (not isinstance(num_repetitions, int)) or num_repetitions <= 0:
        raise ValueError('num_repetitions should be a positive integer')
    if (not isinstance(num_trees, int)) or num_trees <= 0:
        raise ValueError('num_trees should be a positive integer')
    if (not isinstance(max_num_comparisons, int)) or max_num_comparisons <= 0:
        raise ValueError('max_num_comparisons should be a positive integer')
    if (not isinstance(max_num_iterations, int)) or max_num_iterations <= 0:
        raise ValueError('max_num_iterations should be a positive integer')
    if (not isinstance(min_energy_variation, float)) or min_energy_variation < 0:
        raise ValueError('min_energy_variation should be a positive float')

    return cy_kmeans(data, num_centers, distance_b, initialization_b,
                     algorithm_b, num_repetitions, num_trees,
                     max_num_comparisons, max_num_iterations,
                     min_energy_variation, verbose)


def kmeans_quantize(data, centers, distance="l2", algorithm="LLOYD",
                    num_trees=3, max_num_comparisons=100, verbose=False):
    """
    Project the given data to the corresponding cluster indices.

    Parameters
    ----------
    data : [N, D] `float32/float64` `ndarray`
        Input data to be projected.
    centers : [K, D] `float32/float64` `ndarray`
        Clusters centers (must be same data type as data)
    distance : `str`, optional
        Distance to be used ["l1", "l2"]. Default : "l2"
    algorithm : `str`, optional
        One of "LLOYD", "ELKAN", or "ANN". LLOYD is the standard Lloyd
        algorithm (similar to expectation maximisation). ELKAN is a
        faster version of LLOYD using triangular inequalities to cut
        down significantly the number of sample-to-center
        comparisons. ANN is the same as Lloyd, but uses an approximated
        nearest neighbours (ANN) algorithm to accelerate the
        sample-to-center comparisons. The latter is particularly
        suitable for very large problems.
    num_trees : `int`, optional
        The number of trees int the randomized KD-Tree forest (for "ANN").
    max_num_comparisons : `int`, optional
        Maximum number of sample-to-center comparisons when searching
        for the closest center (for "ANN").
    verbose : `bool`, optional
        Outputs information of the converging process

    Returns
    -------
    assignments : [N,] `uint32` `ndarray`
        Assignments to the clusters centers.
    """
    assert isinstance(data, np.ndarray)
    assert isinstance(centers, np.ndarray)
    assert isinstance(verbose, bool)
    if data.ndim != 2:
        raise ValueError('Data should be a 2-D matrix')

    if data.dtype not in [np.float32, np.float64]:
        raise ValueError('Data should be float32 or float64')

    if centers.dtype != data.dtype or centers.shape[1] != data.shape[1]:
        raise ValueError('Data and centers should be of the same dtype and '
                         'with the same number of columns')

    assert centers.shape[0] > 0
    assert data.shape[0] > 0

    distance_b = distance.encode()
    if distance_b not in distance_type.keys():
        raise ValueError('distance field invalid')

    algorithm_b = algorithm.encode()
    if algorithm_b not in algorithm_type.keys():
        raise ValueError('algorithm field invalid')

    if (not isinstance(num_trees, int)) or num_trees <= 0:
        raise ValueError('num_trees should be a positive integer')
    if (not isinstance(max_num_comparisons, int)) or max_num_comparisons <= 0:
        raise ValueError('max_num_comparisons should be a positive integer')

    return cy_kmeans_quantize(data, centers, distance_b, algorithm_b, num_trees,
                              max_num_comparisons, verbose)
