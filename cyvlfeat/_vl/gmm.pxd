# -*- coding: utf-8 -*-
# Author: Alexis Mignon <alexis.mignon@probayes.com>


from .host cimport vl_size, vl_type
from .kmeans cimport VlKMeans

cdef extern from "vl/gmm.h":

    ctypedef enum VlGMMInitialization:
        VlGMMKMeans, VlGMMRand, VlGMMCustom

    ctypedef struct VlGMM:
        pass

    VlGMM *vl_gmm_new (vl_type dataType, vl_size dimension,
                        vl_size numComponents)
    void vl_gmm_delete (VlGMM *self)


    double vl_gmm_cluster(VlGMM *self, void *data, vl_size numData)

    void vl_gmm_set_means(VlGMM *self,  void *means)
    void vl_gmm_set_covariances(VlGMM *self, void* covariances)
    void vl_gmm_set_priors(VlGMM *self, void* priors)

    void vl_gmm_set_num_repetitions(VlGMM *self, vl_size numRepetitions)
    void vl_gmm_set_max_num_iterations(VlGMM *self, vl_size maxNumIterations)
    void vl_gmm_set_verbosity(VlGMM *self, int verbosity)
    void vl_gmm_set_initialization(VlGMM *self, VlGMMInitialization init)
    void vl_gmm_set_kmeans_init_object(VlGMM *self, VlKMeans *kmeans)
    void vl_gmm_set_covariance_lower_bounds(VlGMM *self, double* bounds)
    void vl_gmm_set_covariance_lower_bound(VlGMM *self, double bound)

    void* vl_gmm_get_means(VlGMM *self)
    void* vl_gmm_get_covariances(VlGMM *self)
    void* vl_gmm_get_priors(VlGMM *self)
    void* vl_gmm_get_posteriors(VlGMM *self)

    vl_size vl_gmm_get_num_clusters(VlGMM *self)
    vl_type vl_gmm_get_data_type(VlGMM *self)
    vl_size vl_gmm_get_max_num_iterations(VlGMM *self)
    vl_size vl_gmm_get_num_repetitions(VlGMM *self)
    VlGMMInitialization vl_gmm_get_initialization(VlGMM *self)

    double* vl_gmm_get_covariance_lower_bounds(VlGMM *self)
