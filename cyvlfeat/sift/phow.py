import math
import numpy as np
import matplotlib.colors as colors
import scipy.ndimage
from . import dsift


def phow(image, verbose=False, fast=True, sizes=(4, 6, 8, 10), step=2, color='gray',
         float_descriptors=False, magnif=6, window_size=1.5, contrast_threshold=0.005):
    global frames, descrs

    # Extract the features
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # standardize the image
    color_lower = color.lower()
    set1 = set(color_lower.split(''))
    set2 = set('gray'.split(' '))
    if set1 == set2:
        num_channels = 1

    if image.shape[2] > 1:
        image = rgb2gray(image)

    else:
        num_channels = 3
        if image.shape[2] == 1:
            np.concatenate((image, image, image), axis=2)
        elif color_lower == 'rgb':
            pass
        elif color_lower == 'opponent':

            """Note that the mean differs from the standard definition of opponent
              space and is the regular intensity (for compatibility with
              the contrast thresholding).
              Note also that the mean is added pack to the other two
              components with a small multipliers for monochromatic
              regions."""

            mu = 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]
            alpha = 0.01
            image = np.concatenate((mu, (image[:, :, 0] - image[:, :, 1]) / math.sqrt(2) + alpha * mu,
                                    image[:, :, 0] + image[:, :, 1] - 2 * image[:, :, 2] / math.sqrt(6) + alpha * mu),
                                   axis=2)
        elif color_lower == 'hsv':
            image = colors.rgb_to_hsv(image)
        else:
            color_lower = 'hsv'
            image = colors.rgb_to_hsv(image)
            print ('Color space not recognized, defaulting to HSV color space. \n')

    if verbose:
        print ('Color space: {} \n'.format(color))
        print ('Image size: {}x{} \n'.format(image.shape[0], image.shape[1]))
        print ('Sizes: [{} {} {} {}] \n'.format(sizes[0], sizes[1], sizes[2], sizes[3]))

    for si in range(1, len(sizes)):
        off = math.floor(1.0 + 3.0 / 2.0 * (max(sizes) - sizes[si]))

        # smooth the image to the appropriate scale based on the size of the SIFT bins
        sigma = sizes[si] / magnif
        ims = scipy.ndimage.filters.gaussian_filter(input, sigma)
        infinity = float("inf")
        f = {}
        d = {}

        # extract dense SIFT features from all channels
        for k in range(1, num_channels):
            f, d = dsift.dsift(ims[k, :, :], window_size, verbose, fast, float_descriptors, step, size=sizes[si],
                               bounds=[off, off, infinity, infinity])

        # remove low contrast descriptors note that for color descriptors the V component is thresholded'''

        if color_lower == 'gray' or 'opponent':
            contrast = f[1][2, :]
        elif color_lower == 'rgb':
            m = (f[0][2, :], f[1][2, :], f[2][2, :])
            contrast = np.mean(m, axis=1)
        else:
            color_lower = 'hsv'
            contrast = f[2][2, :]

        frames = []
        descrs = []
        for k in range(1, num_channels):
            d[k][:, d[k].contrast < contrast_threshold] = 0

        param2 = ((sizes[si]) * np.ones(1, (f[1][1]).shape))
        frames[si] = np.row_stack(f[1][(0, 1), :], param2)
        descrs[si] = np.concatenate((d[:]), axis=0)

    frames = np.asarray(frames)
    descrs = np.asarray(descrs)

    return frames, descrs
