import numpy as np


def flatmap(maps):
    """
    Flatten a tree, assigning the label of the root to each node.

    Parameters
    ----------
    maps : [H, W] `float64` `ndarray`.
        Array of the same size of image.
        `maps` as returned by `quickshift`.
        `flatmap` labels each tree of the forest contained in `map`.


    Returns
    -------
    labels : contains the linear index of the root node in `map`.

    clusters : contains a label between 1 and the number of clusters.
    
    Example
    -------
    >>> import numpy as np
    >>> from cyvlfeat.quickshift.quickshift import quickshift
    >>> from cyvlfeat.quickshift.flatmap import flatmap
    >>> from cyvlfeat.test_util import lena
    >>> img = lena().astype(np.float32)
    >>> maps, gaps, estimate = quickshift(img,kernel_size=2,max_dist=10)
    >>> labels, clusters = flatmap(maps)
    """
    maps_shape = maps.shape[0] * maps.shape[1]
    root = np.ones(maps_shape, dtype=np.float64)
    maps_re = maps.reshape(maps_shape)

    # follow the parents list to the root nodes (where nothing changes)
    while 1:
        for i in range(maps_shape):
            val = int(maps_re[i] - 1)
            root[i] = maps_re[val]
        if np.array_equal(maps_re, root):
            break
        # Copy back to maps_re.
        maps_re[:] = root[:]

    labels = maps_re.reshape((maps.shape[0], maps.shape[1]))
    C = np.unique(labels, return_inverse=True)
    clusters = np.array(C[1], dtype=np.float64)
    clusters += 1

    return labels, clusters

