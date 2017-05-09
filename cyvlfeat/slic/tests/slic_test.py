from cyvlfeat.slic.slic import slic
import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.test_util import lena

img = lena().astype(np.float32)


def test_slic_dimension():
    segment = slic(img, region_size=10, regularizer=10)
    assert segment.shape[0] == img.shape[0]
    assert segment.shape[1] == img.shape[1]


def test_slic_segment():
    segment = slic(img, region_size=10, regularizer=10)
    assert_allclose(segment[-4:-1, -1], [2702, 2703, 2703],
                    rtol=1e-3)
    assert_allclose(segment[0:3, -1], [51, 51, 51],
                    rtol=1e-3)
    assert_allclose(segment[0:3, 1], [0, 0, 0],
                    rtol=1e-3)
    assert_allclose(segment[-4:-1, 1], [2600, 2600, 2600],
                    rtol=1e-3)

