from cyvlfeat.vlad.vlad import vlad
import numpy as np
from numpy.testing import assert_allclose

def get_fake_data(dtype=np.float32):
    x = np.random.randn(256, 128)
    mu = np.random.randn(16, 128)
    assignments = np.random.uniform(size=(256, 16))
    assignments = assignments * (1 / assignments.sum(axis=1, keepdims=True))

    return x, mu, assignments

# the helper function is convert from vlfeat's vl_test_vlad
# note that MATLAB uses column-major order while numpy uses row-major order
# tranpose the input array when uses this function
def simple_vlad(x, mu, assign):
    enc = []
    for i in range(assign.shape[0]):
        enc.append(np.matmul(x, assign[i, :].transpose()) - assign[i, :].sum() * mu[:, i])
    enc = np.concatenate(enc)

    return enc

def test_basic():
    for dtype in [np.float32, np.float64]:
        x = np.array([[1], [2], [3]]).astype(dtype)
        mu = np.zeros(shape=(3, 1), dtype=dtype)
        assignments = np.eye(3, dtype=dtype)
        observed_enc = vlad(x, mu, assignments, unnormalized=True)
        expected_enc = np.array([1, 2, 3]).astype(dtype)
        assert_allclose(expected_enc, observed_enc)

def test_rand():
    for dtype in [np.float32, np.float64]:
        x, mu, assignments = get_fake_data(dtype)
        observed_enc = vlad(x, mu, assignments, unnormalized=True)
        expected_enc = simple_vlad(x.transpose(), mu.transpose(), assignments.transpose())
        assert_allclose(observed_enc, expected_enc)

def test_norm():
    for dtype in [np.float32, np.float64]:
        x, mu, assignments = get_fake_data(dtype=dtype)
        expected_enc = simple_vlad(x.transpose(), mu.transpose(), assignments.transpose())
        expected_enc = expected_enc / np.linalg.norm(expected_enc)
        observed_enc = vlad(x, mu, assignments)
        assert_allclose(observed_enc, expected_enc)

def test_sqrt():
    for dtype in [np.float32, np.float64]:
        x, mu, assignments = get_fake_data(dtype=dtype)
        expected_enc = simple_vlad(x.transpose(), mu.transpose(), assignments.transpose())
        expected_enc = np.sign(expected_enc) * np.sqrt(np.abs(expected_enc))
        expected_enc = expected_enc / np.linalg.norm(expected_enc)
        observed_enc = vlad(x, mu, assignments, square_root=True)
        assert_allclose(observed_enc, expected_enc)

def test_individual():
    for dtype in [np.float32, np.float64]:
        x, mu, assignments = get_fake_data(dtype=dtype)
        expected_enc = simple_vlad(x.transpose(), mu.transpose(), assignments.transpose())
        expected_enc = expected_enc.reshape((-1, x.shape[1]))
        expected_enc = expected_enc * (1 / np.sqrt(np.sum(expected_enc**2, axis=1, keepdims=True)))
        expected_enc = expected_enc.flatten()
        observed_enc = vlad(x, mu, assignments, unnormalized=True, normalize_components=True)
        assert_allclose(expected_enc, observed_enc)

def test_mass():
    for dtype in [np.float32, np.float64]:
        x, mu, assignments = get_fake_data(dtype=dtype)
        expected_enc = simple_vlad(x.transpose(), mu.transpose(), assignments.transpose())
        expected_enc = expected_enc.reshape((-1, x.shape[1]))
        expected_enc = expected_enc * np.transpose(1 / assignments.sum(axis=0, keepdims=True))
        expected_enc = expected_enc.flatten()
        observed_enc = vlad(x, mu, assignments, unnormalized=True, normalize_mass=True)
        assert_allclose(expected_enc, observed_enc)

def test_vlad_dimension():
    N = 1000
    K = 512
    D = 128

    x = np.random.uniform(size=(N, D)).astype(np.float32)
    means = np.random.uniform(size=(K, D)).astype(np.float32)
    assignments = np.random.uniform(size=(N, K)).astype(np.float32)
    enc = vlad(x, means, assignments)

    expected = K * D
    observed = len(enc)
    assert (expected == observed)
