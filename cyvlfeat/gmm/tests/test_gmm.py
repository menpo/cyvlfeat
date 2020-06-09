# Author: Alexis Mignon <alexis.mignon@probayes.com>
import numpy as np
import pytest
from numpy.testing import assert_allclose
from cyvlfeat.gmm import gmm

np.random.seed(1)
X = np.random.randn(1000, 2)
X[500:] *= (2, 3)
X[500:] += (4, 4)


def test_gmm_2_clusters_rand_init():
    means, covars, priors, LL, posteriors = gmm(X, n_clusters=2)

    assert_allclose(LL, -4341.0, atol=0.1)
    assert_allclose(priors, [0.5, 0.5], atol=0.1)
    assert_allclose(posteriors[0], [0.0, 1.0], atol=0.1)
    assert_allclose(means, [[4, 4], [0, 0]], atol=0.2)


def test_gmm_2_clusters_kmeans_init():
    means, covars, priors, LL, posteriors = gmm(X, n_clusters=2,
                                                init_mode='kmeans')

    assert_allclose(LL, -4341.0, atol=0.1)
    assert_allclose(priors, [0.5, 0.5], atol=0.1)
    assert_allclose(posteriors[0], [0.0, 1.0], atol=0.1)
    assert_allclose(means, [[4, 4], [0, 0]], atol=0.2)


def test_gmm_2_clusters_custom_init_fail():
    with pytest.raises(ValueError):
        _ = gmm(X, n_clusters=2, init_mode='custom')

