from __future__ import division
from cyvlfeat.kmeans import kmeans
import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.test_util import lena


def test_kmeans_float():
    num_data = 50
    num_centers = 4
    dimension = 4
    noise_level = 0.1

    centers = np.random.random_integers(0, 10, (num_centers, dimension)).astype(np.float32)
    data = np.empty((num_data, dimension), dtype=np.float32)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers, found_assignments = kmeans(data, num_centers, initialization="PLUSPLUS")

    assert found_centers.dtype == np.float32
    assert found_centers.shape == (num_centers, dimension)

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]


def test_kmeans_double():
    num_data = 50
    num_centers = 4
    dimension = 4
    noise_level = 0.1

    centers = np.random.random_integers(0, 10, (num_centers, dimension)).astype(np.float64)
    data = np.empty((num_data, dimension), dtype=np.float64)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers, found_assignments = kmeans(data, num_centers, initialization="PLUSPLUS")

    assert found_centers.dtype == np.float64
    assert found_centers.shape == (num_centers, dimension)

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]