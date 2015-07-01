from cyvlfeat.sift.dsift import dsift
from cyvlfeat.sift.sift import sift
import numpy as np
from scipy.misc import lena


img = lena().astype(np.float32)


def test_dsift_non_float_descriptors():
    i = img.copy()
    frames, descriptors = dsift(i, float_descriptors=False)
    assert descriptors.dtype == np.uint8


def test_dsift_float_descriptors():
    i = img.copy()
    frames, descriptors = dsift(i, float_descriptors=True)
    assert descriptors.dtype == np.float32


def test_sift_float_descriptors():
    i = img.copy()
    frames = sift(i, verbose=True)
    assert frames.shape[0] == 728
