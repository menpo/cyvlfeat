from __future__ import division
from cyvlfeat.hog import hog
import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.test_util import lena

        
img = lena().astype(np.float32) / 255.
half_img = img[:, :256]


def test_hog_cell_size_32_uoctti():
    i = img.copy()
    output = hog(i, 32)
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 31)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.104189, 0.056746],
                              [0.051333, 0.03721]]), rtol=1e-4)


def test_hog_cell_size_32_uoctti_4_orientations():
    i = img.copy()
    output = hog(i, 32, n_orientations=4)
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 16)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.158388,  0.085595],
                              [0.078716,  0.062816]]), rtol=1e-5)


def test_hog_cell_size_32_uoctti_non_square():
    i = half_img.copy()
    output = hog(i, 32)
    assert output.dtype == np.float32
    assert output.shape == (16, 8, 31)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.163912, 0.167787],
                              [0.086294, 0.079365]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs():
    i = img.copy()
    output = hog(i, 32, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 36)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.139408, 0.093407],
                              [0.070996, 0.065033]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs_4_orientations():
    i = img.copy()
    output = hog(i, 32, n_orientations=4, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 16)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.2,      0.154738],
                              [0.109898, 0.108115]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs_non_square():
    i = half_img.copy()
    output = hog(i, 32, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 8, 36)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.2,       0.2],
                              [0.144946,  0.192144]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs_bilinear_interpolation():
    i = img.copy()
    output = hog(i, 32, variant='DalalTriggs', bilinear_interpolation=True)
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 36)
    assert_allclose(output[:2, -2:, 0],
                    np.array([[0.01523,  0.017774],
                              [0.012941, 0.012733]]), rtol=1e-4)
