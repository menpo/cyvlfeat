import math
import numpy as np
from matplotlib import colors as colors
import scipy.ndimage
from . import dsift
from ..utils import conversion as conv


def phow(image, verbose=False, fast=True, sizes=(4, 6, 8, 10), step=2, color='gray',
         float_descriptors=False, magnif=6, window_size=1.5, contrast_threshold=0.005):
    # Standardize the image: The following block assumes that the user input
    # for argument color has somewhat more priority than
    # actual color space of I.
    # That is why the conversions are according to the value of variable 'color'
    # irrespective of actual color space to which I belongs.

    color_lower = color.lower()
    I = image.copy()

    # case where user inputs, color ='gray' and I is also grayscale.
    if color_lower == 'gray':
        num_channels = 1

        # case where user inputs, color ='gray' but I belongs to RGB space.
        if I.ndim == 3 and I.shape[2] > 1:
            I = conv.rgb2gray(I)
    else:
        num_channels = 3

        # case where user inputs, color ='rgb'or 'hsv'or 'opponent' but I is grayscale.
        if I.shape[2] == 1:
            I = np.dstack([I, I, I])

        # case where user inputs, color ='rgb' and I also belongs to RGB space.
        elif color_lower == 'rgb':
            pass

        # case where user inputs, color ='opponent' and I belongs to RGB space.
        elif color_lower == 'opponent':

            # Note that the mean differs from the standard definition of opponent
            # space and is the regular intensity (for compatibility with
            # the contrast thresholding).
            # Note also that the mean is added pack to the other two
            # components with a small multipliers for monochromatic
            # regions.

            alpha = 0.01
            I = np.concatenate(
                (conv.rgb2gray(I), (I[:, :, 0] - I[:, :, 1]) / math.sqrt(2) + alpha * conv.rgb2gray(I),
                 I[:, :, 0] + I[:, :, 1] - 2 * I[:, :, 2] / math.sqrt(6) + alpha * conv.rgb2gray(I)),
                axis=2)
        # case when user inputs, color ='hsv' and I belongs to RGB space.
        elif color_lower == 'hsv':
            I = colors.rgb_to_hsv(I)
        else:
            # case when user inputs, color ='hsv' and I belongs to RGB space.
            color_lower = 'hsv'
            I = colors.rgb_to_hsv(I)
            print('Color space not recognized, defaulting to HSV color space.')

    if verbose:
        print('Color space: {}'.format(color))
        print('I size: {}x{}'.format(I.shape[0], I.shape[1]))
        print('Sizes: [{} {} {} {}]'.format(sizes[0], sizes[1], sizes[2], sizes[3]))

    for si in range(1, len(sizes)):
        off = math.floor(1.0 + 3.0 / 2.0 * (max(sizes) - sizes[si]))

        # smooth I to the appropriate scale based on the size of the SIFT bins
        sigma = sizes[si] / magnif
        ims = scipy.ndimage.filters.gaussian_filter(I, sigma)

        # extract dense SIFT features from all channels
        # temp_results = []
        temp_arr = []
        data = ims.copy()
        for k in range(num_channels):

            # The third dimension of an image matrix represent the no. of channels that are present.
            # In Matlab, size(I) returns: 256 x256 which is same as the result returned by python's I.shape
            # where I is the numpy array of image. In Matlab, size(I,3) returns 1 for a grayscale
            # image but in Python, I.shape[2] raises an error -> tuple index out of range, simply because
            # there is no third channel. For RGB images I.shape[2] returns 3. The below if-else is a fix
            # for that.
            if ims.ndim == 2:
                # Since it is Grayscale, we'd pass whole array (Dsift accepts only 2D arrays.)
                smoothed_image = data

            elif ims.ndim == 3:
                # Since it has 3 channels, i.e. could be split into 3 different channels(2D array) one by one.
                smoothed_image = data[:, :, k]
            else:
                raise ValueError('Image array not defined')

            temp_results = dsift(smoothed_image, step=step, size=sizes[si],
                                 bounds=np.array([off, off, image.shape[0] - 1, image.shape[1] - 1]),
                                 fast=fast, float_descriptors=float_descriptors, verbose=verbose)
            temp_arr.append(temp_results)

        d = [x for (x, y) in temp_arr]
        f = [y for (x, y) in temp_arr]

        if color_lower == 'gray':
            contrast = f[0][1, :]
        elif color_lower == 'opponent':
            contrast = f[0][2, :]
        elif color_lower == 'rgb':
            m = (f[0][2, :], f[1][2, :], f[2][2, :])
            contrast = np.mean(m, axis=1)
        else:
            color_lower = 'hsv'
            contrast = f[2][2, :]

        frames = []
        descrs = []

        # remove low contrast descriptors note that for color descriptors the V component is thresholded

        for k in range(num_channels):
            for i, val in enumerate(contrast):
                if val < contrast_threshold:
                    d[k][i] = 0

        dim2 = contrast.shape[0]
        param2 = (sizes[si]) * np.ones((1, dim2))
        frames.append(np.row_stack((f[0], param2)))

        # for Grayscale
        # for RGB and others, concatenate all elements of d along 0th axis.
        if color_lower == 'gray':
            descrs.append(d)
        else:
            descrs.append(np.concatenate(d, axis=0))

    frames = np.asarray(frames)
    descriptors = np.asarray(descrs)

    return frames, descriptors
