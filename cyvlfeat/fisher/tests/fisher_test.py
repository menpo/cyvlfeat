from cyvlfeat.fisher import fisher
import numpy as np
from numpy.testing import assert_allclose


def test_fisher_dimension():
    N = 1000
    K = 512
    D = 128

    x = np.random.uniform(size=(N, D)).astype(np.float32)
    means = np.random.uniform(size=(K, D)).astype(np.float32)
    covariances = np.random.uniform(size=(K, D)).astype(np.float32)
    priors = np.random.uniform(size=(K,)).astype(np.float32)
    priors /= np.linalg.norm(priors)
    enc = fisher(x, means, covariances, priors, verbose=True)

    expected = 2 * K * D
    observed = len(enc)
    assert(expected == observed)


def _compute_posteriors(diff, covars, prior):
    q = (diff ** 2).sum(-1)  # Mahalanobis distances to cluster centers
    # Compute the log probability of samples knowing each cluster
    q = -0.5 * (q + np.log(covars).sum(-1) +
                diff.shape[1] * np.log(2 * np.pi))
    # Compute the joint probability of xi and j-th cluster
    # p(xi, cj) = p(xi|cj) * p(cj)
    q += np.log(prior)
    # Normalize values to avoid numerical problems
    qmax = q.max(-1)
    q -= qmax[:, None]
    # Compute the posteriors
    q = np.exp(q)
    # normalize
    q /= q.sum(-1)[:, np.newaxis]
    return q


def _fisher_encoding(X, means, covars, priors):
    diff = (X[:, None] - means[None]) / np.sqrt(covars)
    q = _compute_posteriors(diff, covars, priors)
    u = (diff * q[..., np.newaxis]).mean(0) / np.sqrt(priors)[:, np.newaxis]
    v = ((diff ** 2 - 1) * q[..., np.newaxis]).mean(0) / \
        np.sqrt(2 * priors)[:, np.newaxis]
    return np.hstack([u.ravel(), v.ravel()])


def test_fisher_encoding():
    X = np.random.randn(1000, 2)
    X[500:] *= (2, 3)
    X[500:] += (4, 4)

    mu = np.asarray(
        [[0., 0.], [4., 4.]]
    )

    sigma2 = np.asarray(
        [[1.0, 1.0], [2.0, 3.0]]
    )

    prior = np.asarray(
        [0.6, 0.4]
    )

    expected_enc = _fisher_encoding(X, mu, sigma2, prior)
    observed_enc = fisher(X, mu, sigma2, prior, verbose=True)

    assert_allclose(expected_enc, observed_enc, rtol=1e-4)

    observed_enc = fisher(X.astype("float32"),
                          mu.astype("float32"),
                          sigma2.astype("float32"),
                          prior.astype("float32"), verbose=True)

    assert_allclose(expected_enc, observed_enc, rtol=1e-4)
    assert observed_enc.dtype == "float32"
