import numpy as np
from .cyvlad import cy_vlad


def vlad(x, means, assignments, unnormalized=False, square_root=False,
         normalize_components=False, normalize_mass=False, verbose=False):
    r"""
    Computes the VLAD encoding of the vectors ``x`` relative to
    cluster centers ``means`` and vector-to-cluster soft assignments
    ``assignments``.

    Parameters
    ----------
    x : [n_samples, n_features]  `float32` or `float64` `ndarray`
        One row per data vector (e.g. a SIFT descriptor)
    means :  [n_clusters, n_features]  `float32` or `float64` `ndarray`
        One row per kMeans cluster.
    assignments: [n_samples, n_features]  `float32` or `float64` `ndarray`
        No. of columns = No. of clusters
        No. of rows = No. of rows of X
    unnormalized : `bool`, optional
        If ``True``, no overall normalization is applied to the return
        vector.
    square_root : `bool`, optional
        If ``True``, the signed square root function is applied to the return
        vector before normalization.
    normalize_components : `bool`, optional
        If ``True``, the part of the encoding corresponding to each
        cluster is individually normalized.
    normalize_mass : `bool`, optional
        If ``True``, each component is re-normalized by the mass
        of data vectors assigned to it. If ``normalized_components`` is
        also selected, this has no effect.
    verbose: `bool`, optional
        If ``True``, be verbose.

    Returns
    -------
    enc : [k, ] `float32` or `float64` `ndarray`
        A vector of size equal to the product of
        ``k = the n_data_dimensions * n_clusters``.
        
    Examples:
    --------
    >>> from cyvlfeat.vlad.vlad import vlad
    >>> import numpy as np
    >>> N = 1000
    >>> K = 512
    >>> D = 128
    >>> x = np.random.uniform(size=(N, D)).astype(np.float32)
    >>> means = np.random.uniform(size=(K, D)).astype(np.float32)
    >>> assignments = np.random.uniform(size=(N, K)).astype(np.float32)
    >>> enc = vlad(x, means, assignments)
    """
    # check for None
    if x is None or means is None or assignments is None:
        raise ValueError('A required input is None')

    # validate the KMeans parameters
    D = means.shape[1]  # the feature dimensionality
    K = means.shape[0]  # the number of KMeans centers
    # N = x.shape[1] is the number of samples

    # Check one dimension only.
    if x.shape[1] != D:
        raise ValueError('x and means do not have the same dimensionality')

    if assignments.shape[1] != K:
        raise ValueError('assignments has an unexpected shape')

    x = np.ascontiguousarray(x)
    means = np.ascontiguousarray(means)
    assignments = np.ascontiguousarray(assignments)
    result = cy_vlad(x, means, assignments, np.int32(unnormalized),
                     np.int32(square_root), np.int32(normalize_components),
                     np.int32(normalize_mass),
                     np.int32(verbose))

    return result
