from cyvlfeat.fisher import fisher
import numpy as np
from numpy.testing import assert_allclose


def test_fisher_dimension():
    N = 1000
    K = 512
    D = 128
    
    x = np.random.uniform(size=(D, N)).astype(np.float32)
    means = np.random.uniform(size=(D, K)).astype(np.float32)
    covariances = np.random.uniform(size=(D, K)).astype(np.float32)
    priors = np.random.uniform(size=(K,)).astype(np.float32)
    priors /= np.linalg.norm(priors)
    enc = fisher(x, means, covariances, priors, verbose=True)
    
    expected = 2 * K * D
    observed = len(enc)
    assert(expected == observed)


def test_fisher_encoding():
    N = 21
    K = 3
    D = 5
    x = np.zeros(D * N, dtype=np.float32)
    for i in range(D * N):
        x[i] = i
    x = x.reshape(D, N)

    mu = np.zeros(D * K, dtype=np.float32)
    for i in range(D * K):
        mu[i] = i
    mu = mu.reshape(D, K)
    
    sigma2 = np.ones((D, K), dtype=np.float32)
    prior = (1.0 / K) * np.ones(K, dtype=np.float32)

    observed_enc = fisher(x, mu, sigma2, prior, verbose=True)

    # expected result obtained from running vl_fisher_encode from a C program
    expected_enc = np.array([0.000000000, 0.000000000, 0.000000000,
                             0.000000000, 0.000000000, 0.000000000,
                             0.000000000, 0.000000000, 0.000000000,
                             0.000000000, 70.519210815, 70.519210815,
                             70.519210815, 70.519210815, 70.519210815,
                             -0.058321182, -0.058321182, -0.058321182,
                             -0.058321182, -0.058321182, -0.058321182,
                             -0.058321182, -0.058321182, -0.058321182,
                             -0.058321182, 3073.876220703, 3073.876220703,
                             3073.876220703, 3073.876220703, 3073.876220703],
                            dtype=np.float32)
    assert_allclose(expected_enc, observed_enc)
