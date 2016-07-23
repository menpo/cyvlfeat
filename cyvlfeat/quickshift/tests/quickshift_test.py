import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.quickshift.quickshift import quickshift
from cyvlfeat.test_util import lena

img = lena().astype(np.float32)


def test_quickshift_medoid_maps():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10, medoid=True)
    assert maps.shape == (512, 512)
    assert_allclose(maps[0:5, 0], [514., 1026., 1026., 1538., 1538.],
                    rtol=1e-3)


def test_quickshift_medoid_gaps():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10, medoid=True)
    assert gaps.shape == (512, 512)
    assert_allclose(gaps[0:5, 0], [228071.506, 290406.801, 323886.572, 323339.597,
                                   293392.239], rtol=1e-3)


def test_quickshift_medoid_estimate():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10, medoid=True)
    assert estimate.shape == (512, 512)
    assert_allclose(estimate[0:5, 0], [8.699, 11.0754, 12.350, 12.322, 11.190],
                    rtol=1e-3)


def test_quickshift_quick_maps():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10)
    assert maps.shape == (512, 512)
    assert_allclose(maps[0:5, 0], [2., 514., 1026., 1025., 1537.],
                    rtol=1e-3)


def test_quickshift_quick_gaps():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10)
    assert gaps.shape == (512, 512)
    assert_allclose(gaps[0:6, 3], [1., 1., 1., 1.4142, 1., 2.2360],
                    rtol=1e-3)


def test_quickshift_quick_estimate():
    i = img.copy()
    maps, gaps, estimate = quickshift(i, kernel_size=2, max_dist=10)
    assert estimate.shape == (512, 512)
    assert_allclose(estimate[0:5, 0], [8.699, 11.0754, 12.350, 12.322, 11.190],
                    rtol=1e-3)

