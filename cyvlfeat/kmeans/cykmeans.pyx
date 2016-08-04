# distutils: language = c
# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
import numpy as np
cimport numpy as np
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.mathop cimport *
from cyvlfeat._vl.kmeans cimport *
from cyvlfeat._vl.ikmeans cimport *
from cyvlfeat._vl.hikmeans cimport *
from cyvlfeat.cy_util cimport (py_printf, dtype_from_memoryview,
                               set_python_vl_printf)
from libc.string cimport memcpy
from libc.stdlib cimport malloc


ctypedef fused floats:
    np.float32_t
    np.float64_t


algorithm_type = {
    b'LLOYD': VlKMeansLloyd,
    b'ELKAN': VlKMeansElkan,
    b'ANN': VlKMeansANN
}

algorithm_type_ikmeans = {
    b'LLOYD': VL_IKM_LLOYD,
    b'ELKAN': VL_IKM_ELKAN
}

initialization_type = {
    b'RANDSEL': VlKMeansRandomSelection,
    b'PLUSPLUS': VlKMeansPlusPlus
}

distance_type = {
    b'l1': VlDistanceL1,
    b'l2': VlDistanceL2
}


cpdef cy_kmeans(floats[:, ::1] data, int num_centers, bytes distance,
                bytes initialization, bytes algorithm, int num_repetitions,
                int num_trees, int max_num_comparisons, int max_num_iterations,
                float min_energy_variation, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        VlKMeans* kmeans
        double energy
        int dimension = data.shape[1]
        int num_data = data.shape[0]
        vl_type vl_data_type
        floats[:, :] centers

    dtype = dtype_from_memoryview(data)
    if dtype == np.float32:
        vl_data_type = VL_TYPE_FLOAT
    else:
        vl_data_type = VL_TYPE_DOUBLE
    kmeans = vl_kmeans_new(vl_data_type, distance_type[distance])

    vl_kmeans_set_verbosity(kmeans, verbose)
    vl_kmeans_set_num_repetitions(kmeans, num_repetitions)
    vl_kmeans_set_algorithm(kmeans, algorithm_type[algorithm])
    vl_kmeans_set_initialization(kmeans, initialization_type[initialization])
    vl_kmeans_set_max_num_iterations(kmeans, max_num_iterations)
    vl_kmeans_set_max_num_comparisons(kmeans, max_num_comparisons)
    vl_kmeans_set_num_trees(kmeans, num_trees)

    if min_energy_variation > 0:
        vl_kmeans_set_min_energy_variation(kmeans, min_energy_variation)
        
    if verbose:
        py_printf("kmeans: Initialization = %s\n", initialization)
        py_printf("kmeans: Algorithm = %s\n", algorithm)
        py_printf("kmeans: MaxNumIterations = %llu\n", vl_kmeans_get_max_num_iterations(kmeans))
        py_printf("kmeans: MinEnergyVariation = %f\n", vl_kmeans_get_min_energy_variation(kmeans))
        py_printf("kmeans: NumRepetitions = %llu\n", vl_kmeans_get_num_repetitions(kmeans))
        py_printf("kmeans: data type = %s\n", vl_get_type_name(vl_kmeans_get_data_type(kmeans)))
        py_printf("kmeans: distance = %s\n", vl_get_vector_comparison_type_name(vl_kmeans_get_distance(kmeans)))
        py_printf("kmeans: data dimension = %d\n", dimension)
        py_printf("kmeans: num. data points = %d\n", num_data)
        py_printf("kmeans: num. centers = %d\n", num_centers)
        py_printf("kmeans: max num. comparisons = %d\n", max_num_comparisons)
        py_printf("kmeans: num. trees = %d\n", num_trees)

    energy = vl_kmeans_cluster(kmeans, <void*>&data[0, 0], dimension,
                               num_data, num_centers)

    centers = np.empty((num_centers, dimension), dtype=dtype, order='C')
    memcpy(<void*>&centers[0, 0], vl_kmeans_get_centers(kmeans),
          vl_get_type_size(vl_data_type) * dimension * vl_kmeans_get_num_centers(kmeans))

    vl_kmeans_delete(kmeans)

    return np.asarray(centers)


cpdef cy_kmeans_quantize(floats[:, ::1] data, floats[:, ::1] centers,
                         bytes distance, bytes algorithm, int num_trees,
                         int max_num_comparisons, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        VlKMeans *kmeans
        vl_size dimension = data.shape[1]
        vl_size num_data = data.shape[0]
        vl_size num_centers = centers.shape[0]
        np.uint32_t[::1] assignments = np.empty((num_data,), dtype=np.uint32,
                                                order='C')

    dtype = dtype_from_memoryview(data)
    kmeans = vl_kmeans_new(VL_TYPE_FLOAT if dtype == np.float32 else VL_TYPE_DOUBLE,
                           distance_type[distance])

    vl_kmeans_set_verbosity(kmeans, verbose)
    vl_kmeans_set_algorithm(kmeans, algorithm_type[algorithm])
    vl_kmeans_set_max_num_comparisons(kmeans, max_num_comparisons)
    vl_kmeans_set_num_trees(kmeans, num_trees)
    vl_kmeans_set_centers(kmeans, <void*>&centers[0, 0], dimension, num_centers)

    if algorithm_type[algorithm] == VlKMeansANN:
        vl_kmeans_quantize_ann(kmeans, &assignments[0], NULL, <void*>&data[0, 0],
                               num_data, 0)
    else:
        vl_kmeans_quantize(kmeans, &assignments[0], NULL, <void*>&data[0, 0],
                           num_data)

    vl_kmeans_delete(kmeans)

    return np.asarray(assignments)


cpdef cy_ikmeans(np.uint8_t[:, ::1] data, int num_centers, bytes algorithm,
                 int max_num_iterations, bint verbose):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        VlIKMFilt *ikmf
        int err
        int M = data.shape[1]
        int N = data.shape[0]
        int K = num_centers
        np.int32_t[:, :] centers = np.empty((K, M), dtype=np.int32, order='C')
        np.uint32_t[:] assignments = np.empty((N,), dtype=np.uint32, order='C')

    ikmf = vl_ikm_new(algorithm_type_ikmeans[algorithm])

    vl_ikm_set_verbosity(ikmf, verbose)
    vl_ikm_set_max_niters(ikmf, max_num_iterations)

    vl_ikm_init_rand_data(ikmf, &data[0, 0], M, N, K)

    err = vl_ikm_train(ikmf, &data[0, 0], N)
    if err:
        py_printf("ikmeans: possible overflow!\n")

    memcpy(&centers[0, 0], vl_ikm_get_centers(ikmf),
           sizeof(vl_ikmacc_t) * M * K)

    vl_ikm_push(ikmf, &assignments[0], &data[0, 0], N)

    vl_ikm_delete(ikmf)

    if verbose:
        py_printf("ikmeans: done\n")

    return np.asarray(centers), np.asarray(assignments)


cpdef cy_ikmeans_push(np.uint8_t[:, ::1] data, np.int32_t[:, ::1] centers):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        VlIKMFilt *ikmf = vl_ikm_new(VL_IKM_LLOYD)
        int M = data.shape[1]
        int N = data.shape[0]
        int K = centers.shape[0]
        np.uint32_t[:] assignments = np.empty((N,), dtype=np.uint32, order='C')

    vl_ikm_set_verbosity(ikmf, 0)
    vl_ikm_init(ikmf, <vl_ikmacc_t*>&centers[0, 0], M, K)
    vl_ikm_push(ikmf, &assignments[0], &data[0, 0], N)
    vl_ikm_delete(ikmf)

    return np.asarray(assignments)


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
        np.int32_t[:, ::1] centers
        vl_size node_K, M, k

    node_K = vl_ikm_get_K(node.filter)
    M = vl_ikm_get_ndims(node.filter)

    centers = np.empty((node_K, M), dtype=np.int32, order='C')
    if node_K > 0:
        memcpy(&centers[0, 0], vl_ikm_get_centers(node.filter),
               sizeof(vl_ikmacc_t) * M * node_K)
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
        VlHIKMNode *node
        np.int32_t[:, ::1] centers_view

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

    if node_K > 0:
        centers_view = py_node.centers
        vl_ikm_init(node.filter, <vl_ikmacc_t*>&centers_view[0, 0], M,
                    node_K)
    else:
        vl_ikm_init(node.filter, <vl_ikmacc_t*>NULL, M, node_K)

    if len(py_node.children) > 0:
        assert len(py_node.children) == node_K

        node.children = <VlHIKMNode**>malloc(sizeof(VlHIKMNode*) * node_K)
        for k, child in enumerate(py_node.children):
            node.children[k] = build_vl_node(child, tree)
    return node


cdef VlHIKMTree* python_to_hikm(py_tree, int method_type):
    cdef:
        int K, depth
        VlHIKMTree *tree

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


cpdef cy_hikmeans(np.uint8_t[:, ::1] data, int num_clusters, int num_leaves,
                  int depth, bytes algorithm, bint verbose):
    cdef:
        VlHIKMTree *tree
        int M = data.shape[1]
        int N = data.shape[0]
        int K = num_clusters
        np.uint32_t[:, ::1] assignments

    tree  = vl_hikm_new(algorithm_type_ikmeans[algorithm])

    if verbose:
        py_printf("hikmeans: # dims: %d\n", M)
        py_printf("hikmeans: # data: %d\n", N)
        py_printf("hikmeans: K: %d\n", K)
        py_printf("hikmeans: depth: %d\n", depth)

    vl_hikm_set_verbosity(tree, verbose)
    vl_hikm_init(tree, M, K, depth)
    vl_hikm_train(tree, &data[0,0], N)

    py_tree = hikm_to_python(tree)

    assignments = np.empty((N, depth), dtype=np.uint32, order='C')
    vl_hikm_push(tree, &assignments[0,0], &data[0,0], N)

    if verbose:
        py_printf("hikmeans: done.\n")

    vl_hikm_delete(tree)

    return py_tree, np.asarray(assignments)


cpdef cy_hikmeans_push(np.uint8_t[:, ::1] data, py_tree, bint verbose):
    cdef:
        VlHIKMTree* tree
        int depth
        int N = data.shape[0]
        np.uint32_t[:, ::1] assignments

    tree = python_to_hikm(py_tree, VL_IKM_LLOYD)
    depth = vl_hikm_get_depth(tree)

    if verbose:
        py_printf("vl_hikmeanspush: ndims: %llu K: %llu depth: %d\n",
                  vl_hikm_get_ndims(tree),
                  vl_hikm_get_K(tree),
                  depth)

    assignments = np.empty((N, depth), dtype=np.uint32, order='C')
    vl_hikm_push(tree, &assignments[0, 0], &data[0, 0], N)

    vl_hikm_delete(tree)

    return assignments
