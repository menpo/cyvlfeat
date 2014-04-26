import numpy as np
import cysift


def dsift(image, step=1, size=3, bounds=None, window_size=2, norm=False,
          fast=False, float_descriptors=False, geometry=(4, 4, 8),
          verbose=False):
    # Validate image size
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported')

    # Validate bounds
    if bounds is None:
        bounds = np.array([0, 0, image.shape[0], image.shape[1]])
    if bounds.shape[0] != 4:
        raise ValueError('Bounds must be contain 4 elements.')

    # Validate size
    if isinstance(size, int):
        size = np.array([size, size])
    if size.shape[0] != 2:
        raise ValueError('Size vector must contain exactly 2 elements.')
    for s in size:
        if s < 1:
            raise ValueError('Size must only contain positive integers.')

    # Validate step
    if isinstance(step, int):
        step = np.array([step, step])
    if step.shape[0] != 2:
        raise ValueError('Step vector must contain exactly 2 elements.')
    for s in step:
        if s < 1:
            raise ValueError('Step must only contain positive integers.')

    # Validate window_size
    if not isinstance(window_size, int):
        raise ValueError('Window size must be an integer.')
    if window_size < 0:
        raise ValueError('Window size must be a positive integer.')

    # Validate geometry
    geometry = np.asarray(geometry)
    if geometry.shape[0] != 3:
        raise ValueError('Geometry must contain exactly 3 integer elements.')
    if np.min(geometry) < 1:
        raise ValueError('Geometry must only contain positive integers.')

    geometry = geometry.astype(np.int32)
    step = step.astype(np.int32)
    size = size.astype(np.int32)
    bounds = bounds.astype(np.int32)
    image = image.astype(np.float32)
    frames, descriptors = cysift.dsift(image, step, size, bounds, window_size,
                                       norm, fast, float_descriptors, geometry,
                                       verbose)
    print frames
    print descriptors