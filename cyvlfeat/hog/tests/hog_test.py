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
                    np.array([[0.16276041, 0.13416694],
                              [0.22367628, 0.17556792]]), rtol=1e-5)


def test_hog_cell_size_32_uoctti_4_orientations():
    i = img.copy()
    output = hog(i, 32, n_orientations=4)
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 16)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.23047766, 0.17703892],
                              [0.30189356, 0.25290328]]), rtol=1e-5)


def test_hog_cell_size_32_uoctti_non_square():
    i = half_img.copy()
    output = hog(i, 32)
    assert output.dtype == np.float32
    assert output.shape == (16, 8, 31)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.16276041, 0.13416694],
                              [0.22367628, 0.17556792]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs():
    i = img.copy()
    output = hog(i, 32, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 36)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.2, 0.2],
                              [0.2, 0.2]]))


def test_hog_cell_size_32_dalaltriggs_4_orientations():
    i = img.copy()
    output = hog(i, 32, n_orientations=4, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 16)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.2, 0.2],
                              [0.2, 0.2]]))


def test_hog_cell_size_32_dalaltriggs_non_square():
    i = half_img.copy()
    output = hog(i, 32, variant='DalalTriggs')
    assert output.dtype == np.float32
    assert output.shape == (16, 8, 36)
    assert_allclose(output[:2, :2, 0],
                    np.array([[0.2, 0.2],
                              [0.2, 0.2]]), rtol=1e-5)


def test_hog_cell_size_32_dalaltriggs_bilinear_interpolation():
    i = img.copy()
    output = hog(i, 32, variant='DalalTriggs', bilinear_interpolation=True)
    assert output.dtype == np.float32
    assert output.shape == (16, 16, 36)
    assert_allclose(output[:2, -2:, 0],
                    np.array([[0.082442075, 0.13325043],
                              [0.094600961, 0.090005033]]), rtol=1e-5)
