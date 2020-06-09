from __future__ import division
from cyvlfeat.kmeans import kmeans, kmeans_quantize, ikmeans, ikmeans_push, hikmeans, hikmeans_push
import numpy as np


def set_distance(values1, values2):
    """
    Compare two sets of values and returns the maximum distance between the closest pairs
    Parameters
    ----------
    values1 : NxD set of values
    values2 : MxD set of values

    Returns
    -------
    Distance
    """
    dimension = values1.shape[1]
    assert values2.shape[1] == dimension
    dist = np.sqrt(np.sum(
        np.power(values1.reshape((-1, 1, dimension)) - values2.reshape((1, -1, dimension)), 2),
        axis=2))*1/dimension
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (values1.shape[0], values2.shape[0])
    return max(np.max(np.min(dist, axis=0)), np.max(np.min(dist, axis=1)))


def test_kmeans_float():
    num_data = 50
    num_centers = 4
    dimension = 8
    noise_level = 0.1

    centers = np.random.randint(-40, 40, (num_centers, dimension)).astype(np.float32)
    data = np.empty((num_data, dimension), dtype=np.float32)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS")
    found_assignments = kmeans_quantize(data, found_centers)

    assert found_centers.dtype == np.float32
    assert found_centers.shape == (num_centers, dimension)

    assert found_assignments.dtype == np.uint32
    assert found_assignments.shape == (num_data,)

    dist = set_distance(centers, found_centers)
    assert dist <= noise_level, dist

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]


def test_kmeans_double():
    num_data = 50
    num_centers = 4
    dimension = 8
    noise_level = 0.1

    centers = np.random.randint(-40, 40, (num_centers, dimension)).astype(np.float64)
    data = np.empty((num_data, dimension), dtype=np.float64)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS")
    found_assignments = kmeans_quantize(data, found_centers)

    assert found_centers.dtype == np.float64
    assert found_centers.shape == (num_centers, dimension)

    assert found_assignments.dtype == np.uint32
    assert found_assignments.shape == (num_data,)

    dist = set_distance(centers, found_centers)
    assert dist <= noise_level, dist

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]


def test_kmeans_ANN():
    num_data = 5000
    num_centers = 4
    dimension = 8
    noise_level = 0.1

    centers = np.random.randint(-40, 40, (num_centers, dimension)).astype(np.float32)
    data = np.empty((num_data, dimension), dtype=np.float32)
    for i in range(num_data):
        data[i] = centers[i % num_centers] + np.random.random_sample(dimension)*noise_level

    found_centers = kmeans(data, num_centers, initialization="PLUSPLUS", algorithm="ANN")
    found_assignments = kmeans_quantize(data, found_centers, algorithm="ANN")

    assert found_centers.dtype == np.float32
    assert found_centers.shape == (num_centers, dimension)

    dist = set_distance(centers, found_centers)
    assert dist <= noise_level, dist

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]
    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]


def test_ikmeans():
    num_data = 5000
    num_centers = 40
    dimension = 128
    noise_level = 3

    centers = np.random.randint(0, 200, (num_centers, dimension)).astype(np.uint8)
    data = np.empty((num_data, dimension), dtype=np.uint8)
    for i in range(num_data):
        data[i] = centers[i % num_centers]
    data = data + np.random.randint(0, noise_level, (num_data, dimension)).astype(np.uint8)

    found_centers, found_assignments = ikmeans(data, num_centers)

    assert found_centers.dtype == np.int32
    assert found_centers.shape == (num_centers, dimension)

    assert found_assignments.dtype == np.uint32
    assert found_assignments.shape == (num_data,)

    # Because the initialization is random, these tests does not work all the time. Two clusters might be merged.
    # dist = set_distance(centers.astype(np.float32), found_centers.astype(np.float32))
    # assert dist <= noise_level, dist

    # for i in range(num_centers):
    #     for j in range(num_centers):
    #         if i != j:
    #             assert found_assignments[i] != found_assignments[j]

    # for i in range(num_data):
    #     assert found_assignments[i] == found_assignments[i % num_centers]

    assignments_2 = ikmeans_push(data, found_centers)
    assert np.allclose(found_assignments, assignments_2)


def test_ikmeans_2():
    num_data = 5000
    num_centers = 2
    dimension = 2
    noise_level = 3

    centers = np.array([[0, 0], [50, 100]], dtype=np.uint8)
    data = np.empty((num_data, dimension), dtype=np.uint8)
    for i in range(num_data):
        data[i] = centers[i % num_centers]
    data = data + np.random.randint(0, noise_level, (num_data, dimension)).astype(np.uint8)

    found_centers, found_assignments = ikmeans(data, num_centers)

    assert found_centers.dtype == np.int32
    assert found_centers.shape == (num_centers, dimension)

    assert found_assignments.dtype == np.uint32
    assert found_assignments.shape == (num_data,)

    dist = set_distance(centers.astype(np.float32), found_centers.astype(np.float32))
    assert dist <= noise_level, dist

    for i in range(num_centers):
        for j in range(num_centers):
            if i != j:
                assert found_assignments[i] != found_assignments[j]

    for i in range(num_data):
        assert found_assignments[i] == found_assignments[i % num_centers]

    assignments_2 = ikmeans_push(data, found_centers)
    assert np.allclose(found_assignments, assignments_2)


def test_hikmeans():
    num_data = 5000
    num_centers = 40
    dimension = 128
    noise_level = 3

    centers = np.random.randint(0, 200, (num_centers, dimension)).astype(np.uint8)
    data = np.empty((num_data, dimension), dtype=np.uint8)
    for i in range(num_data):
        data[i] = centers[i % num_centers]
    data = data + np.random.randint(0, noise_level, (num_data, dimension)).astype(np.uint8)

    tree_structure, found_assignments = hikmeans(data, 4, 64)

    assert found_assignments.dtype == np.uint32
    assert found_assignments.shape == (num_data, tree_structure.depth)

    assignments_2 = hikmeans_push(data, tree_structure)
    assert np.allclose(found_assignments, assignments_2)
