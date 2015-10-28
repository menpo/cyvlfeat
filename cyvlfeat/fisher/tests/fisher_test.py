from cyvlfeat.fisher import fisher
import numpy as np

def test_fisher():
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
