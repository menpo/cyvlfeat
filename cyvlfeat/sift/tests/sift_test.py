import os.path as osp

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.misc import ascent
from scipy.spatial.distance import cdist

import cyvlfeat
from cyvlfeat.sift.sift import sift

# actual frames and descriptors are compute by sift from vlfeat binary
# command: sift -vv --frames --descriptors --first-octave -1 ascent.pgm
# where ascent.pgm is saved from `scipy.misc.ascent`

# by default `ascent()` return `int64`, but we need `np.float32`
img = ascent().astype(np.float32)
actual_frames = np.loadtxt(osp.join(osp.dirname(cyvlfeat.__file__), 'data', 'ascent.frame'), dtype=np.float32)
actual_frames = actual_frames[:, [1, 0, 2, 3]]
actual_descriptors = np.loadtxt(osp.join(osp.dirname(cyvlfeat.__file__), 'data', 'ascent.descr'), dtype=np.float32)


@pytest.mark.parametrize('compute_descriptor', [True, False])
def test_sift_shape(compute_descriptor):
    result = sift(img, compute_descriptor=compute_descriptor)
    if compute_descriptor:
        assert result[0].shape[1] == 4
        assert result[1].shape[1] == 128
        assert result[0].shape[0] == result[1].shape[0]
    else:
        assert result.shape[1] == 4


def test_sift_user_defined_frames():
    frames, descriptors = sift(img, frames=actual_frames, first_octave=-1,
                               compute_descriptor=True, float_descriptors=True,
                               norm_thresh=0, verbose=False)
    D2 = cdist(actual_frames, frames)
    index = D2.argmin(axis=1)
    descriptors = descriptors[index, :]
    t1 = np.mean(np.sqrt(np.sum((actual_descriptors - descriptors) ** 2, axis=1)))
    t2 = np.mean(np.sqrt(np.sum(actual_descriptors ** 2, axis=1)))
    err = t1 / t2

    assert err < 0.1


def test_sift_detector():
    f = sift(img, first_octave=-1, peak_thresh=0.01)
    frames = actual_frames.copy()
    # scale the components so that 1 pixel erro in x,y,z is equal to a
    # 10-th of angle
    scale = (20 / np.pi)
    frames[:, -1] = np.mod(frames[:, -1], np.pi * 2) * scale
    f[:, -1] = np.mod(f[:, -1], np.pi * 2) * scale
    D2 = cdist(frames, f)
    d2 = D2.min(axis=1)
    d2.sort()
    err = np.sqrt(d2)
    quant80 = round(0.8 * f.shape[0])

    # check for less than one pixel error at 80% quantile
    assert err[quant80] < 1


def test_sift_non_float_descriptors():
    frames, descriptors = sift(img, compute_descriptor=True)

    assert_allclose(frames[0], [5.28098, 342.099, 1.90079, 5.07138], rtol=1e-3)
    assert_allclose(descriptors[0, :5], [17, 0, 0, 11, 20])
    assert frames.shape[0] == 797


def test_sift_sort_user_defined_scales():
    frames = np.array([[5.28098, 342.099, 1.90079, 5.07138],
                       [6.90227, 242.036, 1.84193, 2.63054],
                       [22.1369, 273.583, 2.06255, 1.56508]])
    new_frames, descriptors = sift(img, frames=frames, compute_descriptor=True)

    assert new_frames.shape[0] == 3
    assert_allclose(new_frames[0], frames[1], rtol=1e-3)
    assert_allclose(new_frames[1], frames[0], rtol=1e-3)
    assert_allclose(new_frames[2], frames[2], rtol=1e-3)
    assert_allclose(descriptors[0, :10], [107, 171, 0, 0, 0, 0, 0, 11, 144, 171])
    assert_allclose(descriptors[1, :10], [17, 0, 0, 11, 20, 1, 16, 73, 95, 11])
    assert_allclose(descriptors[2, :10], [134, 15, 0, 0, 1, 1, 0, 0, 134, 73])


def test_sift_force_orientations():
    frames = np.array([[6.90227, 242.036, 1.84193, 2.63054],
                       [5.28098, 342.099, 1.90079, 5.07138],
                       [22.1369, 273.583, 2.06255, 1.56508]])
    new_frames, descriptors = sift(img, frames=frames, compute_descriptor=True,
                                   force_orientations=True)

    assert new_frames.shape[0] == 4
    assert_allclose(new_frames[0], [6.90227, 242.036, 1.84193, 2.63051], rtol=1e-3)
    assert_allclose(descriptors[0, :10], [107, 171, 0, 0, 0, 0, 0, 11, 144, 171])
