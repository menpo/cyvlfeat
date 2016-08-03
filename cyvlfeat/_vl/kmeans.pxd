# Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
# Copyright (C) 2013 Andrea Vedaldi and David Novotny.
# All rights reserved.

# This file is modified from part of the VLFeat library and is made available
# under the terms of the BSD license.
from .host cimport vl_size, vl_index, vl_bool, vl_type, vl_uint32
from .mathop cimport (VlFloatVectorComparisonFunction,
                      VlDoubleVectorComparisonFunction, VlVectorComparisonType)


cdef extern from "vl/kmeans.h":
    
    cdef enum _VlKMeansAlgorithm:
        VlKMeansLloyd = 0,       # Lloyd algorithm
        VlKMeansElkan = 1,       # Elkan algorithm
        VlKMeansANN = 2          # Approximate nearest neighbors
    ctypedef _VlKMeansAlgorithm VlKMeansAlgorithm

    cdef enum _VlKMeansInitialization:
        VlKMeansRandomSelection = 0,  # Randomized selection
        VlKMeansPlusPlus = 1          # Plus plus randomized selection
    ctypedef _VlKMeansInitialization VlKMeansInitialization

    cdef struct _VlKMeans:
        vl_type dataType           # Data type.
        vl_size dimension          # Data dimensionality.
        vl_size numCenters         # Number of centers.
        vl_size numTrees           # Number of trees in forest when using ANN-kmeans.
        vl_size maxNumComparisons  # Maximum number of comparisons when using ANN-kmeans.

        VlKMeansInitialization initialization  # Initialization algorithm.
        VlKMeansAlgorithm algorithm            # Clustering algorithm.
        VlVectorComparisonType distance        # Distance.
        vl_size maxNumIterations               # Maximum number of refinement iterations.
        double minEnergyVariation              # Minimum energy variation.
        vl_size numRepetitions                 # Number of clustering repetitions.
        int verbosity                          # Verbosity level.

        void *centers                          # Centers
        void *centerDistances                  # Centers inter-distances.

        double energy                          # Current solution energy.
        VlFloatVectorComparisonFunction floatVectorComparisonFn
        VlDoubleVectorComparisonFunction doubleVectorComparisonFn
    ctypedef _VlKMeans VlKMeans

    VlKMeans* vl_kmeans_new(vl_type dataType, VlVectorComparisonType distance)
    VlKMeans* vl_kmeans_new_copy(VlKMeans *kmeans)
    void vl_kmeans_delete(VlKMeans *self)


    void vl_kmeans_reset(VlKMeans *self)

    double vl_kmeans_cluster(VlKMeans *self,
                             void *data,
                             vl_size dimension,
                             vl_size numData,
                             vl_size numCenters)

    void vl_kmeans_quantize(VlKMeans *self,
                            vl_uint32 *assignments,
                            void *distances,
                            void *data,
                            vl_size numData)

    void vl_kmeans_quantize_ann(VlKMeans *self,
                                vl_uint32 *assignments,
                                void *distances,
                                void *data,
                                vl_size numData,
                                vl_size iteration )

    void vl_kmeans_set_centers(VlKMeans *self,
                               void *centers,
                               vl_size dimension,
                               vl_size numCenters)

    void vl_kmeans_init_centers_with_rand_data(VlKMeans *self,
                                               void *data,
                                               vl_size dimensions,
                                               vl_size numData,
                                               vl_size numCenters)

    void vl_kmeans_init_centers_plus_plus(VlKMeans *self,
                                          void *data,
                                          vl_size dimensions,
                                          vl_size numData,
                                          vl_size numCenters)

    double vl_kmeans_refine_centers(VlKMeans *self,
                                    void *data,
                                    vl_size numData)

    inline vl_type vl_kmeans_get_data_type(VlKMeans *self)
    inline VlVectorComparisonType vl_kmeans_get_distance(VlKMeans *self)

    inline VlKMeansAlgorithm vl_kmeans_get_algorithm(VlKMeans *self)
    inline VlKMeansInitialization vl_kmeans_get_initialization(VlKMeans *self)
    inline vl_size vl_kmeans_get_num_repetitions(VlKMeans *self)

    inline vl_size vl_kmeans_get_dimension(VlKMeans *self)
    inline vl_size vl_kmeans_get_num_centers(VlKMeans *self)

    inline int vl_kmeans_get_verbosity(VlKMeans *self)
    inline vl_size vl_kmeans_get_max_num_iterations(VlKMeans *self)
    inline double vl_kmeans_get_min_energy_variation(VlKMeans *self)
    inline vl_size vl_kmeans_get_max_num_comparisons(VlKMeans *self)
    inline vl_size vl_kmeans_get_num_trees(VlKMeans *self)
    inline double vl_kmeans_get_energy(VlKMeans *self)
    inline void* vl_kmeans_get_centers(VlKMeans *self)
    
    inline void vl_kmeans_set_algorithm(VlKMeans *self,
                                        VlKMeansAlgorithm algorithm)
    inline void vl_kmeans_set_initialization(
            VlKMeans *self, VlKMeansInitialization initialization)
    inline void vl_kmeans_set_num_repetitions(VlKMeans *self,
                                              vl_size numRepetitions)
    inline void vl_kmeans_set_max_num_iterations(VlKMeans *self,
                                                 vl_size maxNumIterations)
    inline void vl_kmeans_set_min_energy_variation(VlKMeans *self,
                                                   double minEnergyVariation)
    inline void vl_kmeans_set_verbosity(VlKMeans *self, int verbosity)
    inline void vl_kmeans_set_max_num_comparisons(VlKMeans *self,
                                                  vl_size maxNumComparisons)
    inline void vl_kmeans_set_num_trees(VlKMeans *self, vl_size numTrees)
