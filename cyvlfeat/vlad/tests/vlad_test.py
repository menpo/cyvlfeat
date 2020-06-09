from __future__ import division
from cyvlfeat.vlad.vlad import vlad
import numpy as np
from numpy.testing import assert_allclose


def test_vlad_dimension():
    N = 1000
    K = 512
    D = 128

    x = np.random.uniform(size=(D, N)).astype(np.float32)
    means = np.random.uniform(size=(D, K)).astype(np.float32)
    assignments = np.random.uniform(size=(K, N)).astype(np.float32)
    enc = vlad(x, means, assignments)

    expected = K * D
    observed = len(enc)
    assert (expected == observed)


def test_vlad_encoding():
    N = 21
    K = 3
    D = 5
    x = np.zeros(D * N, dtype=np.float32)

    # X has one column per data vector
    for i in range(D * N):
        x[i] = i
    x = x.reshape(D, N)

    # mean has same no. of rows as X and one column per cluster
    mu = np.zeros(D * K, dtype=np.float32)
    for i in range(D * K):
        mu[i] = i
    mu = mu.reshape(D, K)

    # same no. of rows as no. of clusters and as many columns as X
    assign = np.zeros(K * N, dtype=np.float32)
    for i in range(K * N):
        assign[i] = i
    assign = assign.reshape(K, N)
    fraction = K * N
    assign *= (1 / fraction)

    observed_enc = vlad(x, mu, assign)

    # expected result obtained from running vl_vlad_encode from a C program
    expected_enc = np.array([0.27231345, 0.27231345, 0.27231348, 0.27231345, 0.27231342, 0.25836566,
                             0.25836569, 0.25836569, 0.25836569, 0.25836572, 0.24308957, 0.2430896,
                             0.24308957, 0.24308956, 0.24308957], dtype=np.float32)
    assert_allclose(expected_enc, observed_enc)
