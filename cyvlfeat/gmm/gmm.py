# Author: Alexis Mignon <alexis.mignon@probayes.com>
from .cygmm import cy_gmm
import numpy as np


def gmm(X, n_clusters=10, max_num_iterations=100, covariance_bound=None,
        init_mode='rand', init_priors=None, init_means=None, init_covars=None,
        n_repetitions=1, verbose=False):
    """Fit a Gaussian mixture model

    Parameters
    ----------
    X : [n_samples, n_features] `float32/float64` `ndarray`
        The data to be fit. One data point per row.
    n_clusters : `int`, optional
        Number of output clusters.
    max_num_iterations : `int`, optional
        The maximum number of EM iterations.
    covariance_bound : `float` or `ndarray`, optional
        A lower bound on the value of the covariance. If a float is given
        then the same value is given for all features/dimensions. If an
        array is given it should have shape [n_features] and give the
        lower bound for each feature.
    init_mode: {'rand', 'kmeans', 'custom'}, optional
        The initialization mode:

          - rand: Initial mean positions are randomly  chosen among
                  data samples
          - kmeans: The K-Means algorithm is used to initialize the cluster
                    means
          - custom: The intial parameters are provided by the user, through
                    the use of ``init_priors``, ``init_means`` and
                    ``init_covars``. Note that if those arguments are given
                    then the ``init_mode`` value is always considered as
                    ``custom``

    init_priors : [n_clusters,] `ndarray`, optional
        The initial prior probabilities on each components
    init_means : [n_clusters, n_features] `ndarray`, optional
        The initial component means.
    init_covars : [n_clusters, n_features] `ndarray`, optional
        The initial diagonal values of the covariances for each component.
    n_repetitions : `int`, optional
        The number of times the fit is performed. The fit with the highest
        likelihood is kept.
    verbose : `bool`, optional
        If ``True``, display information about computing the mixture model.

    Returns
    -------
    means : [n_clusters, n_features] `ndarray`
        The means of the components
    covars : [n_clusters, n_features] `ndarray`
        The diagonal elements of the covariance matrix for each component.
    priors : [n_clusters] `ndarray`
        The prior probability of each component
    ll : `float`
        The found log-likelihood of the input data w.r.t the fitted model
    posteriors : [n_samples, n_clusters] `ndarray`
        The posterior probability of each cluster w.r.t each data points.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if X.shape[0] == 0:
        raise ValueError('X should contain at least one row')
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains Nans or Infs.")

    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError(
            'n_clusters {} must be a positive integer smaller than the '
            'number of data points {}'.format(n_clusters, n_samples)
        )

    if max_num_iterations < 0:
        raise ValueError('max_num_iterations must be non negative')
    if n_repetitions <= 0:
        raise ValueError('n_repetitions must be a positive integer')
    if init_mode not in {'rand', 'custom', 'kmeans'}:
        raise ValueError("init_mode must be one of {'rand', 'custom', 'kmeans'")

    # Make sure we have the correct types
    X = np.ascontiguousarray(X)
    if X.dtype not in [np.float32, np.float64]:
        raise ValueError('Input data matrix must be of type float32 or float64')

    if covariance_bound is not None:
        covariance_bound = np.asarray(covariance_bound, dtype=np.float)

    if init_priors is not None:
        init_priors = np.require(init_priors, requirements='C', dtype=X.dtype)
        if init_priors.shape != (n_clusters,):
            raise ValueError('init_priors does not have the correct size')
    if init_means is not None:
        init_means = np.require(init_means, requirements='C', dtype=X.dtype)
        if init_means.shape != (n_clusters, n_features):
            raise ValueError('init_means does not have the correct size')
    if init_covars is not None:
        init_covars = np.require(init_covars, requirements='C', dtype=X.dtype)
        if init_covars.shape != (n_clusters, n_features):
            raise ValueError('init_covars does not have the correct size')

    all_inits = (init_priors, init_means, init_covars)
    if any(all_inits) and not all(all_inits):
        raise ValueError('Either all or none of init_priors, init_means and '
                         'init_covars must be set.')

    if init_mode == "custom" and not all(all_inits):
        raise ValueError('init_mode==custom implies that all initial '
                         'parameters are given')

    return cy_gmm(X, n_clusters, max_num_iterations, init_mode.encode('utf8'),
                  n_repetitions, int(verbose),
                  covariance_bound=covariance_bound, init_priors=init_priors,
                  init_means=init_means, init_covars=init_covars)
