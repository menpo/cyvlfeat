from math import sqrt

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.misc import ascent
from skimage.filters import gaussian
from skimage.util import img_as_float32

from cyvlfeat.sift.dsift import dsift
from cyvlfeat.sift.sift import sift

img = ascent().astype(np.float32)


def test_dsift_slow_fast():
    # bin size in pixels
    bin_size = 4
    # bin size / keypoint scale
    magnif = 3
    scale = bin_size / magnif
    window_size = 5

    img_smooth = gaussian(img, sigma=sqrt(scale ** 2 - 0.25))
    _, d = dsift(img_smooth, size=bin_size, step=10,
                 window_size=window_size, float_descriptors=True)
    _, d_ = dsift(img_smooth, size=bin_size, step=10,
                  window_size=window_size, float_descriptors=True,
                  fast=True)
    err = np.std(d_ - d) / np.std(d)

    assert err < 0.1


@pytest.mark.parametrize('window_size', [5, 7.5, 10, 12.5, 20])
def test_dsift_sift(window_size):
    bin_size = 4
    magnif = 3
    scale = bin_size / magnif
    img_smooth = gaussian(img, sigma=sqrt(scale ** 2 - 0.25))
    f, d = dsift(img_smooth, size=bin_size,
                 step=10, window_size=window_size,
                 float_descriptors=True)
    num_keypoints = f.shape[0]
    f_ = np.column_stack([f, np.ones(shape=(num_keypoints,)) * scale, np.zeros(shape=(num_keypoints,))])
    f_, d_ = sift(img, magnification=magnif, frames=f_,
                  first_octave=-1, n_levels=5, compute_descriptor=True,
                  float_descriptors=True, window_size=window_size)
    err = np.std(d - d_) / np.std(d)

    assert err < 0.1


def test_dsift_non_float_descriptors():
    _, descriptors = dsift(img, float_descriptors=False)
    assert descriptors.dtype == np.uint8


def test_dsift_float_descriptors():
    _, descriptors = dsift(img, float_descriptors=True)
    assert descriptors.dtype == np.float32


def test_dsift_steps():
    # Step 3 in Y-Direction, 4 in X-Direction
    frames, descriptors = dsift(img, step=[3, 4])

    assert frames.shape[0] == 21168
    assert_allclose(frames[:3], [[4.5, 4.5],
                                 [4.5, 8.5],
                                 [4.5, 12.5]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, :10], [99, 0, 0, 0, 0, 0, 150, 24, 56, 0])


def test_dsift_windowsize():
    frames, descriptors = dsift(img, window_size=3)

    assert frames.shape[0] == 253009
    assert_allclose(frames[:3], [[4.5, 4.5],
                                 [4.5, 5.5],
                                 [4.5, 6.5]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, :10], [99, 0, 0, 0, 0, 0, 157, 24, 52, 0],
                    rtol=1e-3)


def test_dsift_norm():
    frames, descriptors = dsift(img, norm=True)

    assert frames.shape[-1] == 3
    assert frames.shape[0] == 253009
    print(frames)
    assert_allclose(frames[:3], [[4.5, 4.5, 0.2953],
                                 [4.5, 5.5, 0.2471],
                                 [4.5, 6.5, 0.2115]],
                    rtol=1e-3)
    assert_allclose(descriptors[0, :10], [99, 0, 0, 0, 0, 0, 150, 24, 56, 0],
                    rtol=1e-3)
