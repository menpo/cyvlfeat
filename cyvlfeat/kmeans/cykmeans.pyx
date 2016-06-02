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
from cyvlfeat._vl.ikmeans cimport *
from cyvlfeat._vl.hikmeans cimport *
from libc.stdio cimport printf
from libc.string cimport memcpy
from libc.stdlib cimport malloc

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
        printf("%s", "\n")

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


algorithm_type_ikmeans = {b'LLOYD': VL_IKM_LLOYD, b'ELKAN': VL_IKM_ELKAN}


cpdef cy_ikmeans(np.uint8_t[:,:] data, int num_centers, bytes algorithm, int max_num_iterations, bint verbose):

    cdef:
        VlIKMFilt* ikmf
        int M, N, K
        int err
        np.ndarray[int, ndim=2, mode='c'] centers
        np.ndarray[unsigned int, ndim=1, mode='c'] assignments

    M = data.shape[1]
    N = data.shape[0]
    K = num_centers

    ikmf = vl_ikm_new(algorithm_type_ikmeans[algorithm])

    vl_ikm_set_verbosity(ikmf, verbose)
    vl_ikm_set_max_niters(ikmf, max_num_iterations)

    vl_ikm_init_rand_data(ikmf, &data[0,0], M, N, K)

    err = vl_ikm_train(ikmf, &data[0,0], N)
    if err:
        printf("%s", "ikmeans: possible overflow!\n")

    centers = np.empty((K, M), dtype=np.int32, order='C')
    memcpy(&centers[0,0], vl_ikm_get_centers(ikmf), sizeof(vl_ikmacc_t) * M * K)

    assignments = np.empty((N,), dtype=np.uint32, order='C')
    vl_ikm_push(ikmf, &assignments[0], &data[0,0], N)

    vl_ikm_delete(ikmf)

    if verbose:
        printf("%s", "ikmeans: done\n")

    return centers, assignments


cpdef cy_ikmeans_push(np.uint8_t[:,:] data, np.int32_t[:,:] centers):
    cdef:
        VlIKMFilt* ikmf
        int M, N, K
        np.ndarray[unsigned int, ndim=1, mode='c'] assignments

    M = data.shape[1]
    N = data.shape[0]
    K = centers.shape[0]

    ikmf = vl_ikm_new(VL_IKM_LLOYD)
    vl_ikm_set_verbosity(ikmf, 0)
    vl_ikm_init(ikmf, <vl_ikmacc_t*>&centers[0,0], M, K)

    assignments = np.empty((N,), dtype=np.uint32, order='C')
    vl_ikm_push(ikmf, &assignments[0], &data[0,0], N)

    vl_ikm_delete(ikmf)

    return assignments


class PyHIKMNode:
    def __init__(self):
        self.centers = None
        self.children = []


class PyHIKMTree(PyHIKMNode):
    def __init__(self, K, depth):
        PyHIKMNode.__init__(self)
        self.K = K
        self.depth = depth


cdef build_py_node(py_node, VlHIKMNode *node):
    cdef:
        np.ndarray[int, ndim=2, mode='c'] centers
        int node_K, M

    node_K = vl_ikm_get_K(node.filter)
    M = vl_ikm_get_ndims(node.filter)

    centers = np.empty((node_K, M), dtype=np.int32, order='C')
    if node_K>0:
        memcpy(&centers[0,0], vl_ikm_get_centers(node.filter), sizeof(vl_ikmacc_t) * M * node_K)
    py_node.centers = centers

    if node.children:
        for k in range(node_K):
            msub = PyHIKMNode()
            build_py_node(msub, node.children[k])
            py_node.children.append(msub)


cdef hikm_to_python(VlHIKMTree* tree):
    py_tree = PyHIKMTree(vl_hikm_get_K(tree), vl_hikm_get_depth (tree))
    build_py_node(py_tree, tree.root)
    return py_tree


cdef VlHIKMNode* build_vl_node(py_node, VlHIKMTree *tree):
    cdef:
        vl_size M, node_K
        VlHIKMNode* node
        np.uint32_t[:,:] centers_view

    assert isinstance(py_node, PyHIKMNode)

    M = py_node.centers.shape[1]
    node_K = py_node.centers.shape[0]

    assert node_K <= tree.K
    if tree.M == 0:
        tree.M = M
    else:
        assert M == tree.M

    node = <VlHIKMNode*>malloc(sizeof(VlHIKMNode))
    node.filter = vl_ikm_new(tree.method)
    node.children = NULL

    if node_K>0:
        centers_view = py_node.centers.view(np.uint32)
        vl_ikm_init(node.filter, <vl_ikmacc_t*>&centers_view[0,0], M, node_K)
    else:
        vl_ikm_init(node.filter, <vl_ikmacc_t*>NULL, M, node_K)

    if len(py_node.children)>0:
        assert len(py_node.children) == node_K

        node.children = <VlHIKMNode**>malloc(sizeof(VlHIKMNode *) * node_K)
        for (k, child) in enumerate(py_node.children):
            node.children[k] = build_vl_node(child, tree)
    return node


cdef VlHIKMTree* python_to_hikm(py_tree, int method_type):
    cdef:
        int K, depth
        VlHIKMTree* tree

    assert isinstance(py_tree, PyHIKMTree), "Tree must be of class PyHIKMTree"
    K = py_tree.K
    depth = py_tree.depth

    tree = <VlHIKMTree*>malloc(sizeof(VlHIKMTree))
    tree.depth = <vl_size>depth
    tree.K = <vl_size>K
    tree.M = 0  # to be initialized later
    tree.method = method_type
    tree.root = build_vl_node(py_tree, tree)
    return tree


cpdef cy_hikmeans(np.uint8_t[:,:] data, int num_clusters, int num_leaves, int depth, bytes algorithm, bint verbose):
    cdef:
        VlHIKMTree* tree
        int M, N, K
        np.ndarray[unsigned int, ndim=2, mode='c'] assignments

    M = data.shape[1]
    N = data.shape[0]
    K = num_clusters

    tree  = vl_hikm_new(algorithm_type_ikmeans[algorithm])

    if verbose:
        printf("hikmeans: # dims: %d\n", M)
        printf("hikmeans: # data: %d\n", N)
        printf("hikmeans: K: %d\n", K)
        printf("hikmeans: depth: %d\n", depth)

    vl_hikm_set_verbosity(tree, verbose)
    vl_hikm_init(tree, M, K, depth)
    vl_hikm_train(tree, &data[0,0], N)

    py_tree = hikm_to_python(tree)

    assignments = np.empty((N, depth), dtype=np.uint32, order='C')
    vl_hikm_push(tree, &assignments[0,0], &data[0,0], N)

    if verbose:
        printf("%s", "hikmeans: done.\n")

    vl_hikm_delete(tree)

    return py_tree, assignments


cpdef cy_hikmeans_push(np.uint8_t[:,:] data, py_tree, bint verbose):
    cdef:
        VlHIKMTree* tree
        int depth, N
        np.ndarray[unsigned int, ndim=2, mode='c'] assignments

    tree = python_to_hikm(py_tree, VL_IKM_LLOYD)
    depth = vl_hikm_get_depth(tree)
    N = data.shape[0]

    if verbose:
        printf("vl_hikmeanspush: ndims: %d K: %d depth: %d\n",
                vl_hikm_get_ndims(tree),
                vl_hikm_get_K(tree),
                depth)

    assignments = np.empty((N, depth), dtype=np.uint32, order='C')
    vl_hikm_push(tree, &assignments[0,0], &data[0,0], N)

    vl_hikm_delete(tree)

    return assignments
