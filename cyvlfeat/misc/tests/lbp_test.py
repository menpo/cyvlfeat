from cyvlfeat.misc.lbp import lbp
import numpy as np
from numpy.testing import assert_allclose
from cyvlfeat.test_util import lena

img = lena().astype(np.float32)


def test_lbp_histograms_dimensions():
    i = img.copy()
    result = lbp(i, cell_size=200)
    assert result.shape[0] == 2
    assert result.shape[1] == 2
    assert result.shape[2] == 58


def test_lbp_histograms():
    i = img.copy()
    result = lbp(i, cell_size=300)
    # result obtained from running vl_lbp from a C program
    assert_allclose(result[:, :, :],
                    [[[0.12391094, 0.08487658, 0.10579009, 0.15625666, 0.12110113, 0.08651522,
                       0.10110916, 0.10902751, 0.0931019, 0.1432666, 0.14087646, 0.08676995,
                       0.07968471, 0.10241799, 0.11125918, 0.09384136, 0.08991235, 0.08638003,
                       0.07019918, 0.07879133, 0.08337893, 0.12014022, 0.08423576, 0.07791713,
                       0.10960911, 0.09774201, 0.07664397, 0.08685651, 0.12191707, 0.08265419,
                       0.10080042, 0.15547319, 0.12076242, 0.07734945, 0.093639, 0.10460736,
                       0.08661077, 0.14596128, 0.13832623, 0.07620393, 0.075464, 0.10150401,
                       0.09879602, 0.08539315, 0.07967681, 0.08023989, 0.06469885, 0.07515154,
                       0.08807483, 0.10723767, 0.08069197, 0.07049505, 0.11228816, 0.10078386,
                       0.08175401, 0.09329715, 0.36979756, 0.54350281]]]
                    , rtol=1e-3)
