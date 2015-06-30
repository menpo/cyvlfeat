from cyvlfeat.sift.dsift import dsift
from cyvlfeat.sift.sift import sift
import numpy as np
from scipy.misc import lena

def test_dsift_non_float_descriptors():
    img = np.random.random([100, 100])
    frames, descriptors = dsift(img, float_descriptors=False)
    assert descriptors.dtype == np.uint8


def test_dsift_float_descriptors():
    img = np.random.random([100, 100])
    frames, descriptors = dsift(img, float_descriptors=True)
    assert descriptors.dtype == np.float32


def test_sift_float_descriptors():
    img = lena().astype(np.float32)
    frames = sift(img, verbose=True)
    print(frames[0])
    assert frames.shape[0] == 728
