from __future__ import division
from cyvlfeat.kmeans import kmeans, kmeans_quantize
import numpy as np


def same_set(values1, values2, precision):
    """
    Compare two sets of values and check that they are roughly the same
    Parameters
    ----------
    values1 : NxD set of values
    values2 : MxD set of values
    precision : precision allowed for the values to mismatch

    Returns
    -------
    Whether or not each set contains a neighbouring pair in the other set
    """
    dimension = values1.shape[1]
    assert values2.shape[1] == dimension
    dist = np.sqrt(np.sum(
        np.power(values1.reshape((-1, 1, dimension)) - values2.reshape((1, -1, dimension)), 2),
        axis=2))*1/dimension
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (values1.shape[0], values2.shape[1])
    return np.all(np.min(dist, axis=0) < precision) and np.all(np.min(dist, axis=1) < precision)


def test_kmeans_float():
    num_data = 50
    num_centers = 4
    dimension = 4
    noise_level = 0.1

    centers = np.random.random_integers(-20, 20, (num_centers, dimension)).astype(np.float32)
    data = np.empty((num_data, dimension), dtype=np.float32)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS")
    found_assignments = kmeans_quantize(data, found_centers)

    assert found_centers.dtype == np.float32
    assert found_centers.shape == (num_centers, dimension)

    assert same_set(centers, found_centers, 0.1)

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

    centers = np.random.random_integers(-20, 20, (num_centers, dimension)).astype(np.float64)
    data = np.empty((num_data, dimension), dtype=np.float64)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS")
    found_assignments = kmeans_quantize(data, found_centers)

    assert found_centers.dtype == np.float64
    assert found_centers.shape == (num_centers, dimension)

    assert same_set(centers, found_centers, 0.1)

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]


def test_kmeans_ANN():
    num_data = 5000
    num_centers = 4
    dimension = 4
    noise_level = 0.1

    centers = np.random.random_integers(-20, 20, (num_centers, dimension)).astype(np.float32)
    data = np.empty((num_data, dimension), dtype=np.float32)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS", algorithm="ANN")
    found_assignments = kmeans_quantize(data, found_centers, algorithm="ANN")

    assert found_centers.dtype == np.float32
    assert found_centers.shape == (num_centers, dimension)

    assert same_set(centers, found_centers, 0.1)

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]
