import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.misc import face
import cyvlfeat
from cyvlfeat.sift.phow import phow
import pytest


img_rgb = face(gray=False).astype(np.float32)
img_gray = face(gray=True).astype(np.float32)


@pytest.mark.parametrize('color', ['gray', 'rgb', 'hsv', 'opponent'])
def test_phow(color):
    if color == 'gray':
        frames, descriptors = phow(img_gray, color=color)
        assert descriptors.shape[1] == 128
    else:
        frames, descriptors = phow(img_rgb, color=color)
        assert descriptors.shape[1] == 128*3