# -*- coding: utf-8 -*-
# Author: Alexis Mignon <alexis.mignon@probayes.com>
""" Wrapper for the gmm module of vlfeat
"""

from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.gmm cimport *
from libc.stdio cimport printf
from libc.string cimport memcpy
import numpy as np
cimport numpy as np


def cy_gmm(np.ndarray X, int num_clusters=10, int max_num_iterations=100,
          bint verbose=False,
          np.ndarray[np.float64_t, ndim=1] covariance_bound=None,
          np.ndarray init_priors=None,
          np.ndarray init_means=None,
          np.ndarray init_covars=None, init_mode="rand",
          int num_repetitions=1):
    cdef :
        vl_size i

        vl_size dimension
        vl_size numData

        void * initCovariances = NULL
        void * initMeans = NULL
        void * initPriors = NULL
        bint init_all_ok = VL_FALSE

        double covarianceScalarBound = np.nan
        double* covarianceBound = NULL
        void* data = NULL
        float* data_float = NULL
        double* data_double = NULL

        vl_size maxNumIterations = max_num_iterations
        vl_size numRepetitions = num_repetitions
        double LL
        int verbosity = 0
        VlGMMInitialization initialization = VlGMMRand


        vl_type dataType ;

        VlGMM * gmm ;

        np.ndarray out_means
        np.ndarray out_covars
        np.ndarray out_priors

    if X.dtype == "float32":
          dataType = VL_TYPE_FLOAT
    elif X.dtype == "float64":
          dataType = VL_TYPE_DOUBLE
    else:
        raise ValueError("Data type should be 'float32' or 'float64'")

    if X.ndim != 2:
        raise ValueError("Data should have 2 dimensions")

    dimension = X.shape[1]
    numData = X.shape[0]

    if verbose:
        verbosity += 1

    if dimension == 0:
        raise ValueError("X should contain at least one row")

    if num_clusters <= 0 or num_clusters > numData:
        raise ValueError(
          "num_clusters (%i) must be a positive integer smaller than the number "
          "of data points (%i)" % (num_clusters, numData)
        )


    if max_num_iterations < 0:
        raise ValueError("max_num_iterations must be non negative")

    if covariance_bound is not None:
        if covariance_bound.shape[0] == 1:
            covarianceScalarBound = covariance_bound[0]
        else:
            covarianceBound = <np.float64_t*> covariance_bound.data


    if init_priors is not None:
        if init_priors.dtype != X.dtype:
            raise ValueError(
                "init_priors does not have the same dtype as the data X"
            )

        if init_priors.ndim != 1 or init_priors.shape[0] != num_clusters:
            raise ValueError(
                "init_priors does not have the correct size"
            )

        initPriors = <void*> init_priors.data

    if init_means is not None:
        if init_means.dtype != X.dtype:
            raise ValueError(
                "init_means does not have the same dtype as the data X"
            )

        if init_means.ndim != 1 or init_means.shape[0] != dimension:
          raise ValueError(
                "init_means does not have the correct size"
          )

        initMeans = <void*> init_priors.data

    if init_covars is not None:
        if init_covars.dtype != X.dtype:
            raise ValueError(
                "init_means does not have the same dtype as the data X"
            )

        if init_covars.ndim != 1 or init_covars.shape[0] != dimension:
            raise ValueError(
                "init_means does not have the correct size"
            )

        initCovariances = <void*> init_covars.data

    if init_mode == "rand":
        initialization = VlGMMRand
    elif init_mode == "custom":
        initialization = VlGMMCustom
    elif init_mode == "kmeans":
        initialization = VlGMMKMeans
    else:
        raise ValueError("Invalid value for init_mode")

    if num_repetitions < 1:
        raise ValueError("num_repetitions is not larger or equal to 1.")

  # -----------------------------------------------------------------
  #                                                  Do the job
  # --------------------------------------------------------------

    data = <void*>X.data

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains Nans or Infs.")

    if (
        init_priors is not None or
        init_means is not None or
        init_covars is not None
    ):
        if (init_priors is None or init_means is None or init_covars is None):
          raise ValueError("All or none of init_priors, init_means, "
                           "init_covars must be set.")
        else:
            init_all_ok = True

    if init_mode == "custom" and not init_all_ok:
        raise ValueError("custom init_mode implies that all initial "
                         "parameters are given")

    if init_all_ok and not init_mode == "custom":
        init_mode = "custom"
        initialization = VlGMMCustom

    gmm = vl_gmm_new (dataType, dimension, num_clusters)
    vl_gmm_set_verbosity (gmm, verbosity)
    vl_gmm_set_num_repetitions (gmm, num_repetitions)
    vl_gmm_set_max_num_iterations (gmm, max_num_iterations)
    vl_gmm_set_initialization (gmm, initialization)

    if not np.isnan(covarianceScalarBound):
        vl_gmm_set_covariance_lower_bound(gmm, covarianceScalarBound)

    if covarianceBound:
        vl_gmm_set_covariance_lower_bounds(gmm, covarianceBound)

    if initPriors:
        vl_gmm_set_priors(gmm, initPriors)

    if initMeans:
        vl_gmm_set_means(gmm, initMeans) ;

    if initCovariances:
        vl_gmm_set_covariances(gmm, initCovariances)

    if verbosity:
        printf("vl_gmm: initialization = %s\n", <char*>init_mode) ;
        printf("vl_gmm: maxNumIterations = %d\n", vl_gmm_get_max_num_iterations(gmm)) ;
        printf("vl_gmm: numRepetitions = %d\n", vl_gmm_get_num_repetitions(gmm)) ;
        printf("vl_gmm: data type = %s\n", vl_get_type_name(vl_gmm_get_data_type(gmm))) ;
        printf("vl_gmm: data dimension = %d\n", dimension) ;
        printf("vl_gmm: num. data points = %d\n", numData) ;
        printf("vl_gmm: num. Gaussian modes = %d\n", num_clusters) ;
        printf("vl_gmm: lower bound on covariance = [") ;


        if dimension < 3:
            for i in range(dimension):
                printf(" %f", vl_gmm_get_covariance_lower_bounds(gmm)[i]) ;

        else:
            printf(" %f %f ... %f",
                    vl_gmm_get_covariance_lower_bounds(gmm)[0],
                    vl_gmm_get_covariance_lower_bounds(gmm)[1],
                    vl_gmm_get_covariance_lower_bounds(gmm)[dimension-1]) ;

        printf("%s", "]\n") ;


    # -------------------------------------------------------------- */
    #                                                 Clustering */
    # -------------------------------------------------------------- */

    LL = vl_gmm_cluster(gmm, data, numData) ;

    # copy centers
    out_means = np.zeros((num_clusters, dimension), X.dtype)
    out_covars = np.zeros((num_clusters, dimension), X.dtype)
    out_priors = np.zeros(num_clusters, X.dtype)
    #out_posteriors = np.zeros((numData, num_clusters), X.dtype)

    data = <void*> out_means.data
    memcpy(data,
           vl_gmm_get_means (gmm),
           vl_get_type_size (dataType) * dimension *
           vl_gmm_get_num_clusters(gmm))

    data = <void*> out_covars.data
    memcpy (data,
            vl_gmm_get_covariances (gmm),
            vl_get_type_size (dataType) * dimension *
            vl_gmm_get_num_clusters(gmm)) ;

    data = <void*> out_priors.data
    memcpy (data,
            vl_gmm_get_priors (gmm),
            vl_get_type_size (dataType) * vl_gmm_get_num_clusters(gmm)) ;

    vl_gmm_delete(gmm)

    return out_means, out_covars, out_priors
