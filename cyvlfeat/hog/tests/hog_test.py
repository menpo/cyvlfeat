import cyvlfeat
from cyvlfeat.hog import hog
import numpy as np
from numpy.testing import assert_allclose
from scipy.misc import face
from scipy.io import loadmat
import pytest
import os.path as osp

img_gray = face(gray=True).astype(np.float32)
img_rgb = face(gray=False).astype(np.float32)
# HOG features obtained from the MATLAB/Octave API
hog_features = loadmat(osp.join(osp.dirname(cyvlfeat.__file__), 'data', 'hog.mat'))


@pytest.mark.parametrize('cell_size', [8, 15, 24, 32])
@pytest.mark.parametrize('n_orientations', [4, 6, 9, 15])
@pytest.mark.parametrize('variant', ['DalalTriggs', 'UoCTTI'])
@pytest.mark.parametrize('visualize', [True, False])
def test_hog_shape(cell_size, n_orientations, variant, visualize):
    output = hog(img_rgb, cell_size=cell_size,
                 n_orientations=n_orientations,
                 variant=variant, visualize=visualize)
    h = int((img_rgb.shape[0] + cell_size / 2) // cell_size)
    w = int((img_rgb.shape[1] + cell_size / 2) // cell_size)
    if visualize:
        viz_h = 21 * h
        viz_w = 21 * w
    if variant == 'UoCTTI':
        c = 4 + 3 * n_orientations
    else:
        c = 4 * n_orientations
    if visualize:
        assert output[0].dtype == np.float32
        assert output[0].shape == (h, w, c)
        assert output[1].shape == (viz_h, viz_w)
    else:
        assert output.dtype == np.float32
        assert output.shape == (h, w, c)


def test_hog_cell_size_32_uoctti():
    des_gray = hog(img_gray, 32)
    des_rgb = hog(img_rgb, 32)

    assert_allclose(hog_features['hog_des_gray_uoctti'], des_gray, rtol=1e-3)
    assert_allclose(hog_features['hog_des_rgb_uoctti'], des_rgb, rtol=1e-3)


@pytest.mark.parametrize('bilinear_interpolation', [True, False])
def test_hog_cell_size_32_dalaltriggs(bilinear_interpolation):
    des_gray = hog(img_gray, 32, variant='DalalTriggs', bilinear_interpolation=bilinear_interpolation)
    des_rgb = hog(img_rgb, 32, variant='DalalTriggs', bilinear_interpolation=bilinear_interpolation)

    if bilinear_interpolation:
        assert_allclose(hog_features['hog_des_gray_dalaltriggs_bilinear'], des_gray, rtol=1e-4)
        assert_allclose(hog_features['hog_des_rgb_dalaltriggs_bilinear'], des_rgb, rtol=1e-4)
    else:
        assert_allclose(hog_features['hog_des_gray_dalaltriggs'], des_gray, rtol=1e-4)
        assert_allclose(hog_features['hog_des_rgb_dalaltriggs'], des_rgb, rtol=1e-4)

@pytest.mark.parametrize('variant', ['DalalTriggs', 'UoCTTI'])
def test_hog_channel_order(variant):
    img_rgb_chanel_first = img_rgb.transpose([2, 0, 1])
    des_rgb_channel_first, viz_rgb_channel_first = hog(img_rgb_chanel_first, 32, variant=variant,
                                                       channels_first=True, visualize=True)
    des_rgb, viz_rgb = hog(img_rgb, 32, variant=variant, channels_first=False, visualize=True)

    assert des_rgb.shape == des_rgb_channel_first.shape
    assert viz_rgb.shape == viz_rgb_channel_first.shape
    assert_allclose(des_rgb, des_rgb_channel_first)
    assert_allclose(viz_rgb, viz_rgb_channel_first)
