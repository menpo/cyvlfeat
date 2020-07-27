import numpy as np
import pytest
from scipy.misc import ascent, face

from cyvlfeat.quickshift.flatmap import flatmap
from cyvlfeat.quickshift.quickshift import quickshift

img = ascent().astype(np.float64)
img_rgb = face(gray=False).astype(np.float64)


def test_quickshift_rgb():
    result = quickshift(img_rgb,
                        kernel_size=4,
                        max_dist=24)
    assert result[0].dtype == np.float64
    assert result[1].dtype == np.float64
    assert result[2].dtype == np.float64
    assert result[0].shape == img_rgb.shape[:2]
    assert result[1].shape == img_rgb.shape[:2]
    assert result[2].shape == img_rgb.shape[:2]


@pytest.mark.parametrize('kernel_size', [2, 4, 8])
@pytest.mark.parametrize('max_dist', [None, 10, 20, 30])
@pytest.mark.parametrize('medoid', [True, False])
def test_quickshift(kernel_size, max_dist, medoid):
    result = quickshift(img,
                        kernel_size=kernel_size,
                        max_dist=max_dist,
                        medoid=medoid)
    assert result[0].dtype == np.float64
    assert result[1].dtype == np.float64
    assert result[0].shape == img.shape
    assert result[1].shape == img.shape
    if max_dist is not None:
        assert result[2].dtype == np.float64
        assert result[2].shape == img.shape


def test_flatmap():
    maps = quickshift(img, 2, 10)[0]
    labels, clusters = flatmap(maps)

    assert maps.dtype == np.float64
    assert maps.dtype == img.dtype
    assert maps.shape == img.shape
    assert labels.dtype == maps.dtype
    assert labels.shape == maps.shape
    assert clusters.dtype == maps.dtype
    assert clusters.size == maps.size
