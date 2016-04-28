# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
cimport cython
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.mathop cimport *
from cyvlfeat._vl.kmeans cimport *
from libc.stdio cimport printf
from libc.string cimport memcpy

algorithm_type = {b'LLOYD': VlKMeansLloyd, b'ELKAN': VlKMeansElkan, b'ANN': VlKMeansANN}

initialization_type = {b'RANDSEL': VlKMeansRandomSelection, b'PLUSPLUS': VlKMeansPlusPlus}

distance_type = {b'l1': VlDistanceL1, b'l2': VlDistanceL2}

cpdef cy_kmeans(np.ndarray data, int num_centers, bytes distance, bytes initialization,
           bytes algorithm, int num_repetitions, int num_trees, int max_num_comparisons,
           int max_num_iterations, float min_energy_variation, bint verbose):

    cdef:
        VlKMeans* kmeans
        double energy
        int dimension
        int num_data
        vl_type data_type
        np.ndarray centers
        np.uint8_t[:,:] data_view = data.view(np.uint8)
        np.uint8_t[:,:] centers_view

    dimension = data.shape[1]
    num_data = data.shape[0]

    if data.dtype == np.float32:
        data_type = VL_TYPE_FLOAT
    else:
        data_type = VL_TYPE_DOUBLE
    kmeans = vl_kmeans_new(data_type, distance_type[distance])

    vl_kmeans_set_verbosity(kmeans, verbose)
    vl_kmeans_set_num_repetitions(kmeans, num_repetitions)
    vl_kmeans_set_algorithm(kmeans, algorithm_type[algorithm])
    vl_kmeans_set_initialization(kmeans, initialization_type[initialization])
    vl_kmeans_set_max_num_iterations(kmeans, max_num_iterations)
    vl_kmeans_set_max_num_comparisons(kmeans, max_num_comparisons)
    vl_kmeans_set_num_trees(kmeans, num_trees)

    if min_energy_variation>0:
        vl_kmeans_set_min_energy_variation(kmeans, min_energy_variation)
        
    if verbose:
        printf("kmeans: Initialization = %s\n", initialization)
        printf("kmeans: Algorithm = %s\n", algorithm)
        printf("kmeans: MaxNumIterations = %d\n", vl_kmeans_get_max_num_iterations(kmeans))
        printf("kmeans: MinEnergyVariation = %f\n", vl_kmeans_get_min_energy_variation(kmeans))
        printf("kmeans: NumRepetitions = %d\n", vl_kmeans_get_num_repetitions(kmeans))
        printf("kmeans: data type = %s\n", vl_get_type_name(vl_kmeans_get_data_type(kmeans)))
        printf("kmeans: distance = %s\n", vl_get_vector_comparison_type_name(vl_kmeans_get_distance(kmeans)))
        printf("kmeans: data dimension = %d\n", dimension)
        printf("kmeans: num. data points = %d\n", num_data)
        printf("kmeans: num. centers = %d\n", num_centers)
        printf("kmeans: max num. comparisons = %d\n", max_num_comparisons)
        printf("kmeans: num. trees = %d\n", num_trees)
        printf("\n")

    energy = vl_kmeans_cluster(kmeans, &data_view[0, 0], dimension, num_data, num_centers)

    if data_type == VL_TYPE_FLOAT:
        centers = np.empty((num_centers, dimension), dtype=np.float32, order='C')
    else:
        centers = np.empty((num_centers, dimension), dtype=np.float64, order='C')
    centers_view = centers.view(np.uint8)

    memcpy(&centers_view[0,0], vl_kmeans_get_centers (kmeans),
          vl_get_type_size(data_type) * dimension * vl_kmeans_get_num_centers(kmeans))

    vl_kmeans_delete(kmeans)

    return centers


cpdef cy_kmeans_quantize(np.ndarray data, np.ndarray centers, bytes distance, bytes algorithm,
                         int num_trees, int max_num_comparisons, bint verbose):
    cdef:
        VlKMeans* kmeans
        vl_size dimension
        vl_size num_data
        vl_size num_centers
        vl_type data_type
        np.ndarray[unsigned int, ndim=1, mode='c'] assignments
        np.uint8_t[:,:] data_view = data.view(np.uint8)
        np.uint8_t[:,:] centers_view = centers.view(np.uint8)

    dimension = data.shape[1]
    num_data = data.shape[0]
    num_centers = centers.shape[0]

    if data.dtype == np.float32:
        data_type = VL_TYPE_FLOAT
    else:
        data_type = VL_TYPE_DOUBLE
    kmeans = vl_kmeans_new(data_type, distance_type[distance])

    vl_kmeans_set_verbosity(kmeans, verbose)
    vl_kmeans_set_algorithm(kmeans, algorithm_type[algorithm])
    vl_kmeans_set_max_num_comparisons(kmeans, max_num_comparisons)
    vl_kmeans_set_num_trees(kmeans, num_trees)
    vl_kmeans_set_centers(kmeans, &centers_view[0, 0], dimension, num_centers)

    assignments = np.empty((num_data,), dtype=np.uint32, order='C')

    if algorithm_type[algorithm] == VlKMeansANN:
        vl_kmeans_quantize_ann(kmeans, &assignments[0], NULL, &data_view[0, 0], num_data, 0)
    else:
        vl_kmeans_quantize(kmeans, &assignments[0], NULL, &data_view[0, 0], num_data)

    vl_kmeans_delete(kmeans)

    return assignments