# Author: Alexis Mignon <alexis.mignon@probayes.com>
""" Wrapper for the GMM module from vlfeat
"""

from .cygmm import cy_gmm
import numpy as np


def _check_contiguous(arr):
    """Checks that array is C contiguous

    Checks that an array is C contiguous and converts it if needed
    """
    if arr is not None:
        return np.ascontiguousarray(arr)


def gmm(X, num_clusters=10, max_num_iterations=100,
        verbose=False, covariance_bound=None, init_mode="rand",
        init_priors=None, init_means=None, init_covars=None,
        num_repetitions=1):
    """ Fit a Gaussian mixture model

    Parameters
    ----------
    X: array_like of shape [n_samples, n_features]
        The data to be fit. One data point per row

    max_num_iterations: int
        The maximum number of EM iterations

    verbose: bool, optional (default=False)
        If True, displays some informations.

    covariance_bound: float or array_like, optional (default=None)
        A lower bound on the value of the covariance. If a float is given
        then the same value is given for all features/dimensions. If an
        array is given it should have shape [n_features] and give the
        lower bound for each feature.

    init_mode: 'rand', 'kmeans' or 'custom', optional (default='rand')
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

    init_prior: array_like of shape [num_components], optional (default=None)
        The initial prior probabilities on each components

    init_means: array of shape [num_components, n_features], optional
        The initial component means. (default=None)

    init_covar: array of shape [num_components, n_features], optional
        The initial diagonal values of the covariances for each component.
        (default=None)

    num_repetitions: int, optional (default=1)
        The number of times the fit is performed. The fit with the highest
        likelihood is kept.

    Returns
    -------
    priors: ndarray of shape [num_clusters]
        The prior probability of each component
    means: ndarray of shape [num_clusters, n_features]
        The means of the components
    covars: ndarray of shape [num_clusters, n_features]
        The diagonal elements of the covariance matrix for each component.
    ll: float
        The found log-likelihood of the input data w.r.t the fitted model
    posteriors: ndarray of shape [n_samples, num_clusters]
        The posterior probability of each cluster w.r.t each data points.
    """
    X = _check_contiguous(X)

    if covariance_bound is not None:
        try:
            covariance_bound = np.asarray([float(covariance_bound)])
        except TypeError:
            covariance_bound = _check_contiguous(covariance_bound)

    init_priors = _check_contiguous(init_priors)
    init_means = _check_contiguous(init_means)
    init_covars = _check_contiguous(init_covars)

    return cy_gmm(
        X=X,
        num_clusters=num_clusters,
        max_num_iterations=max_num_iterations,
        verbose=verbose,
        covariance_bound=covariance_bound,
        init_mode=init_mode,
        init_priors=init_priors,
        init_means=init_means,
        init_covars=init_covars,
        num_repetitions=num_repetitions
    )
