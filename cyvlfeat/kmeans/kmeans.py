import numpy as np
from .cykmeans import cy_kmeans, algorithm_type, initialization_type, distance_type


def kmeans(data, num_centers, distance="l2", initialization="RANDSEL",
           algorithm="LLOYD", num_repetitions=1, num_trees=3, max_num_comparisons=100,
           max_num_iterations=100, min_energy_variation=0, verbose=False):
    r"""Cluster data using k-means
       Clusters the columns of the
       matrix X in NUMCENTERS centers C using k-means. X may be either
       SINGLE or DOUBLE. C has the same number of rows of X and NUMCENTER
       columns, with one column per center. A is a UINT32 row vector
       specifying the assignments of the data X to the NUMCENTER
       centers.
    
       [C, A, ENERGY] = VL_KMEANS(...) returns the energy of the solution
       (or an upper bound for the ELKAN algorithm) as well.
    
       KMEANS() supports different initialization and optimization
       methods and different clustering distances. Specifically, the
       following options are supported:
    
       Verbose::
         Increase the verbosity level (may be specified multiple times).
    
       Distance:: [L2]
         Use either L1 or L2 distance.
    
       Initialization::
         Use either random data points (RANDSEL) or k-means++ (PLUSPLUS)
         to initialize the centers.
    
       Algorithm:: [LLOYD]
         One of LLOYD, ELKAN, or ANN. LLOYD is the standard Lloyd
         algorithm (similar to expectation maximisation). ELKAN is a
         faster version of LLOYD using triangular inequalities to cut
         down significantly the number of sample-to-center
         comparisons. ANN is the same as Lloyd, but uses an approximated
         nearest neighbours (ANN) algorithm to accelerate the
         sample-to-center comparisons. The latter is particularly
         suitable for very large problems.
    
       NumRepetitions:: [1]
         Number of time to restart k-means. The solution with minimal
         energy is returned.
    
       The following options tune the KD-Tree forest used for ANN
       computations in the ANN algorithm (see also VL_KDTREEBUILD()
       andVL_KDTREEQUERY()).
    
       NumTrees:: [3]
         The number of trees int the randomized KD-Tree forest.
    
       MaxNumComparisons:: [100]
         Maximum number of sample-to-center comparisons when searching
         for the closest center.
    
       MaxNumIterations:: [100]
         Maximum number of iterations allowed for the kmeans algorithm
         to converge.
    
       Example::
         VL_KMEANS(X, 10, 'verbose', 'distance', 'l1', 'algorithm',
         'elkan') clusters the data point X using 10 centers, l1
         distance, and the Elkan's algorithm."""

    assert isinstance(data, np.ndarray)
    assert isinstance(num_centers, int)
    assert isinstance(verbose, bool)
    if data.ndim != 2:
        raise ValueError('Data should be a 2-D matrix')

    if data.dtype not in [np.float32, np.float64]:
        raise ValueError('Data should be float32 or float64')

    if num_centers > data.shape[0]:
        raise ValueError('num_centers should be a positive integer smaller than the number of data points')

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

    return cy_kmeans(data, num_centers, distance_b, initialization_b,
                     algorithm_b, num_repetitions, num_trees, max_num_comparisons,
                     max_num_iterations, min_energy_variation, verbose)
