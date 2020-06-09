import numpy as np
from .cyfisher import cy_fisher


def fisher(x, means, covariances, priors, normalized=False, square_root=False,
           improved=False, fast=False, verbose=False):
    r"""
    Computes the Fisher vector encoding of the vectors ``x`` relative to
    the diagonal covariance Gaussian mixture model with ``means``,
    ``covariances``, and prior mode probabilities ``priors``.

    By default, the standard Fisher vector is computed.

    Parameters
    ----------
    x : [D, N]  `float32` `ndarray`
        One column per data vector (e.g. a SIFT descriptor) descriptor_dim x number_of_features
    means :  [F, N]  `float32` `ndarray`
        One column per GMM component. i.e. descriptor_dim x num_cluster
    covariances :  [F, N]  `float32` `ndarray`
        One column per GMM component (covariance matrices are assumed diagonal,
        hence these are simply the variance of each data dimension). descriptor_dim x num_cluster
    priors :  [F, N]  `float32` `ndarray`
        Equal to the number of GMM components. i.e. a column vector of length num_cluster
    normalized : `bool`, optional
        If ``True``, L2 normalize the Fisher vector.
    square_root : `bool`, optional
        If ``True``, the signed square root function is applied to the return
        vector before normalization.
    improved : `bool`, optional
        If ``True``, compute the improved variant of the Fisher Vector. This is
        equivalent to specifying the ``normalized`` and ``square_root` options.
    fast : `bool`, optional
        If ``True``, uses slightly less accurate computations but significantly
        increase the speed in some cases (particularly with a large number of
        Gaussian modes).
    verbose: `bool`, optional
        If ``True``, print information.

    Returns
    -------
    enc : [k, 1] `float32` `ndarray`
        A vector of size equal to the product of
        ``k = 2 * the n_data_dimensions * n_components``.
    """
    # check for None
    if x is None or means is None or covariances is None or priors is None:
        raise ValueError('A required input is None')

    # validate the gmm parameters
    D = means.shape[0]  # the feature dimensionality
    K = means.shape[1]  # the number of GMM modes
    # N = x.shape[1] is the number of samples
    if covariances.shape[0] != D:
        raise ValueError('covariances and means do not have the same '
                         'dimensionality')
    
    if priors.ndim != 1:
        raise ValueError('priors has an unexpected shape')
    
    if covariances.shape[1] != K or priors.shape[0] != K:
        raise ValueError('covariances or priors does not have the correct '
                         'number of modes')

    if x.shape[0] != D:
        raise ValueError('x and means do not have the same dimensionality')

    return cy_fisher(x, means, covariances, priors,
                     np.int32(normalized), np.int32(square_root),
                     np.int32(improved), np.int32(fast), np.int32(verbose))
