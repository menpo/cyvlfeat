from cyvlfeat.fisher import fisher
import numpy as np

def test_fisher():
    N = 1000
    K = 512
    D = 128
    
    x = np.random.uniform(size=(D,N)).astype(dtype=np.float32)
    means = np.random.uniform(size=(D,K)).astype(dtype=np.float32)
    covariances = np.random.uniform(size=(D,K)).astype(dtype=np.float32)
    priors = np.random.uniform(size=(K,)).astype(dtype=np.float32)
    priors /= np.linalg.norm(priors)
    enc = fisher(x,means,covariances,priors,Verbose=True)
    
    expected = 2*K*D
    observed = len(enc)
    assert(expected==observed)
