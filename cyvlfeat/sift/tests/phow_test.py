import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.datasets import face
import cyvlfeat
from cyvlfeat.sift.phow import phow
import pytest


# only the descriptor size is checked and phow is slow on the full image, so use a crop
img_rgb = np.ascontiguousarray(face(gray=False)[:192, :256]).astype(np.float32)
img_gray = np.ascontiguousarray(face(gray=True)[:192, :256]).astype(np.float32)


@pytest.mark.parametrize('color', ['gray', 'rgb', 'hsv', 'opponent'])
def test_phow(color):
    if color == 'gray':
        frames, descriptors = phow(img_gray, color=color)
        assert descriptors.shape[1] == 128
    else:
        frames, descriptors = phow(img_rgb, color=color)
        assert descriptors.shape[1] == 128*3