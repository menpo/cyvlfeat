from __future__ import division
from cyvlfeat.fisher import fisher
import numpy as np
from numpy.testing import assert_allclose

def test_fisher():
    N = 1000
    K = 512
    D = 128
    
    x = np.random.uniform(size=(D,N))
    means = np.random.uniform(size=(D,K))
    covariances = np.random.uniform(size=(D,K))
    priors = np.linalg.norm(np.random.uniform(size=(K,)))
    enc = fisher(x,means,covariances,priors,Verbose=True)
    
    expected = 2*K*D
    observed = len(enc)
    assert(expected==observed)
