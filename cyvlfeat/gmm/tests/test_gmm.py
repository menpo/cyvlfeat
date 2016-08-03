# -*- coding: utf-8 -*-
# Author: Alexis Mignon <alexis.mignon@probayes.com>
# Date: Thu Jun  2 14:28:08 2016
# Copyright (c) 2016 ProbaYes SAS
"""
"""
import numpy as np
from cyvlfeat.gmm import gmm


def test_gmm():
    np.random.seed(1)
    X = np.random.randn(1000, 2)
    X[500:] *= (2, 3)
    X[500:] += (4, 4)
    means, covars, priors = gmm(X, num_clusters=2)
    np.testing.assert_allclose(priors, [0.5, 0.5], atol=0.1)

    try:
        np.testing.assert_allclose(means, [[0, 0], [4, 4]], atol=0.2)
    except AssertionError:
        np.testing.assert_allclose(means, [[4, 4], [0, 0]], atol=0.2)
