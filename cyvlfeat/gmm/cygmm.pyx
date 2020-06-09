# Author: Alexis Mignon <alexis.mignon@probayes.com>
from cyvlfeat._vl.host cimport *
from cyvlfeat._vl.gmm cimport *
from cyvlfeat.cy_util cimport (dtype_from_memoryview, py_printf,
                               set_python_vl_printf)
from libc.string cimport memcpy
import numpy as np
cimport numpy as np


ctypedef fused floats:
    np.float32_t
    np.float64_t


initialization_type = {
    b'rand': VlGMMRand,
    b'kmeans': VlGMMKMeans,
    b'custom': VlGMMCustom
}

inv_initialization_type = {
    VlGMMRand: b'rand',
    VlGMMKMeans: b'kmeans',
    VlGMMCustom: b'custom'
}


def cy_gmm(floats[:, :] data, int n_clusters, int max_num_iterations,
           bytes init_mode, int num_repetitions, int verbose,
           floats[:] covariance_bound=None, floats[:] init_priors=None,
           floats[:, :] init_means=None, floats[:, :] init_covars=None,):
    # Set the vlfeat printing function to the Python stdout
    set_python_vl_printf()

    cdef:
        vl_size i
        vl_size n_samples = data.shape[0]
        vl_size n_features = data.shape[1]

        floats covar_scalar_bound = np.nan
        floats *covar_arr_bound = NULL

        floats LL = 0
        VlGMMInitialization vl_init_mode

        vl_type vl_data_type
        VlGMM *gmm

        floats[:, ::1] out_means
        floats[:, ::1] out_covars
        floats[::1] out_priors
        floats[:, ::1] out_posteriors

    vl_init_mode = initialization_type[init_mode]

    dtype = dtype_from_memoryview(data)
    if dtype == np.float32:
          vl_data_type = VL_TYPE_FLOAT
    elif dtype == np.float64:
          vl_data_type = VL_TYPE_DOUBLE

    if covariance_bound is not None:
        if covariance_bound.shape[0] == 1:
            covar_scalar_bound = covariance_bound[0]
        else:
            covar_arr_bound = &covariance_bound[0]

    gmm = vl_gmm_new(vl_data_type, n_features, n_clusters)
    vl_gmm_set_verbosity(gmm, verbose)
    vl_gmm_set_num_repetitions(gmm, num_repetitions)
    vl_gmm_set_max_num_iterations(gmm, max_num_iterations)
    vl_gmm_set_initialization(gmm, vl_init_mode)

    if not np.isnan(covar_scalar_bound):
        vl_gmm_set_covariance_lower_bound(gmm, covar_scalar_bound)

    if covar_arr_bound:
        vl_gmm_set_covariance_lower_bounds(gmm, <double*>covar_arr_bound)

    if init_priors is not None:
        vl_gmm_set_priors(gmm, <void*>&init_priors[0])

    if init_means is not None:
        vl_gmm_set_means(gmm, <void*>&init_means[0, 0])

    if init_covars is not None:
        vl_gmm_set_covariances(gmm, <void*>&init_covars[0, 0])

    if verbose > 0:
        mode = inv_initialization_type[vl_gmm_get_initialization(gmm)]
        py_printf("vl_gmm: vl_init_mode = %s\n", <const char *>mode)
        py_printf("vl_gmm: maxNumIterations = %llu\n", vl_gmm_get_max_num_iterations(gmm))
        py_printf("vl_gmm: numRepetitions = %llu\n", vl_gmm_get_num_repetitions(gmm))
        py_printf("vl_gmm: data type = %s\n", vl_get_type_name(vl_gmm_get_data_type(gmm)))
        py_printf("vl_gmm: data n_features = %llu\n", n_features)
        py_printf("vl_gmm: num. data points = %llu\n", n_samples)
        py_printf("vl_gmm: num. Gaussian modes = %d\n", n_clusters)

        py_printf("vl_gmm: lower bound on covariance = [")
        if n_features < 3:
            for i in range(n_features):
                py_printf(" %f", vl_gmm_get_covariance_lower_bounds(gmm)[i])
        else:
            py_printf(" %f %f ... %f",
                      vl_gmm_get_covariance_lower_bounds(gmm)[0],
                      vl_gmm_get_covariance_lower_bounds(gmm)[1],
                      vl_gmm_get_covariance_lower_bounds(gmm)[n_features-1])

        py_printf("%s", "]\n")

    # Clustering .....................................
    LL = vl_gmm_cluster(gmm, <void*>&data[0, 0], n_samples)

    out_means = np.zeros((n_clusters, n_features), dtype=dtype)
    out_covars = np.zeros((n_clusters, n_features), dtype=dtype)
    out_priors = np.zeros(n_clusters, dtype=dtype)
    out_posteriors = np.zeros((n_samples, n_clusters), dtype=dtype)

    memcpy(<void*>&out_means[0, 0],
           vl_gmm_get_means(gmm),
           vl_get_type_size(vl_data_type) * n_features * vl_gmm_get_num_clusters(gmm))

    memcpy(<void*>&out_covars[0, 0],
           vl_gmm_get_covariances(gmm),
           vl_get_type_size(vl_data_type) * n_features * vl_gmm_get_num_clusters(gmm))

    memcpy(<void*>&out_priors[0],
           vl_gmm_get_priors(gmm),
           vl_get_type_size(vl_data_type) * vl_gmm_get_num_clusters(gmm))

    memcpy(<void*>&out_posteriors[0, 0],
           vl_gmm_get_posteriors(gmm),
           vl_get_type_size(vl_data_type) * n_samples * vl_gmm_get_num_clusters(gmm))

    vl_gmm_delete(gmm)

    return (np.asarray(out_means), np.asarray(out_covars),
            np.asarray(out_priors), LL, np.asarray(out_posteriors))
