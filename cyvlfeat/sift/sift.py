import numpy as np
import cyvlfeat.sift.cysift as cysift


def sift(image, n_octaves=-1, n_levels=3,  first_octave=0,  peak_thresh=0,
         edge_thresh=10, norm_thresh=-np.inf,  magnif=3, window_size=2,
         frames=None, force_orientations=False, float_descriptors=False,
         compute_descriptor=False, verbose=False):
    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    image = np.require(image, dtype=np.float32, requirements='F')
    frames = cysift.sift(image, n_octaves, n_levels,
                         first_octave,  peak_thresh,
                         edge_thresh, norm_thresh,  magnif,
                         window_size, frames, force_orientations,
                         float_descriptors, compute_descriptor,
                         verbose)
    return frames
