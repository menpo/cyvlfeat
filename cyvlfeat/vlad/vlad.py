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
    x : [D, N]  `float32` `ndarray`
        One column per data vector (e.g. a SIFT descriptor)
    means :  [F, N]  `float32` `ndarray`
        One column per kMeans cluster.
    assignments: [F, N]  `float32` `ndarray`
        No. of rows = No. of clusters
        No. of columns = No. of columns of X
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
    enc : [k, ] `float32` `ndarray`
        A vector of size equal to the product of
        ``k = the n_data_dimensions * n_clusters``.
        
    Examples:
    --------
    >>> from cyvlfeat.vlad.vlad import vlad
    >>> import numpy as np
    >>> N = 1000
    >>> K = 512
    >>> D = 128
    >>> x = np.random.uniform(size=(D, N)).astype(np.float32)
    >>> means = np.random.uniform(size=(D, K)).astype(np.float32)
    >>> assignments = np.random.uniform(size=(K, N)).astype(np.float32)
    >>> enc = vlad(x, means, assignments)
    """
    # check for None
    if x is None or means is None or assignments is None:
        raise ValueError('A required input is None')

    # validate the KMeans parameters
    D = means.shape[0]  # the feature dimensionality
    K = means.shape[1]  # the number of KMeans centers
    # N = x.shape[1] is the number of samples

    # Check one dimension only.
    if x.shape[0] != D:
        raise ValueError('x and means do not have the same dimensionality')

    if assignments.shape[0] != K:
        raise ValueError('assignments has an unexpected shape')

    result = cy_vlad(x, means, assignments, np.int32(unnormalized),
                     np.int32(square_root), np.int32(normalize_components),
                     np.int32(normalize_mass),
                     np.int32(verbose))

    return result
