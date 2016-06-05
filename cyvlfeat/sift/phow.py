from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import dsift


def phow(image, verbose=False, fast=True, sizes=[4, 6, 8, 10], step=2, color='gray',
         floatdescriptors=False, magnif=6, window_size=1.5, contrastthreshold=0.005):
    # -------------------------------------------------------------------
    #  Parse the arguments
    # -------------------------------------------------------------------
    dsiftOpts = ['norm', 'window_size', window_size]

    if verbose:
        dsiftOpts.append('verbose')

    if fast:
        dsiftOpts.append('fast')

    if floatdescriptors:
        dsiftOpts.append('floatdescriptors')

    dsiftOpts.extend(['step', step])

    # -------------------------------------------------------------------
    # Extract the features
    # -------------------------------------------------------------------
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # standardize the image
    imageSize = [image.shape[0], image.shape[1]]

    color_lower = color.lower()
    gray_lower = 'gray'.lower()
    set1 = set(color_lower.split(''))
    set2 = set(gray_lower.split(' '))
    if (set1 == set2):
        numChannels = 1

    if image.shape[2] > 1:
        image = mpimg.imread(image)
        # print img
        image = rgb2gray(image)
        # print "\n printing grayscale \n"
        # print gray

    else:
        numChannels = 3
        if image.shape[2] == 1:
            np.concatenate((image, image, image), axis=2)
        elif color_lower == 'rgb':
            pass
        elif color_lower == 'opponent':
            '''Note that the mean differs from the standard definition of opponent
              space and is the regular intensity (for compatibility with
              the contrast thresholding).
              Note also that the mean is added pack to the other two
              components with a small multipliers for monochromatic
              regions.'''
            mu = 0.3 * image[0, :, :] + 0.59 * image[1, :, :] + 0.11 * image[2, :, :]
            alpha = 0.01
            image = np.concatenate((mu, (image[0, :, :] - image[1, :, :]) / math.sqrt(2) + alpha * mu,
                                    image[0, :, :] + image[1, :, :] - 2 * image[2, :, :] / math.sqrt(6) + alpha * mu),
                                   axis=2)
        elif color_lower == 'hsv':
            image = colors.rgb_to_hsv(image)
        else:
            color_lower = 'hsv'
            image = colors.rgb_to_hsv(image)
            print 'Color space not recongized, defaulting to HSV color space.'

    if verbose:
        print 'color space ' + color + '\n'
        print 'image size ' + imageSize[0] + imageSize[1]
        print 'sizes [' + '{d[0]} {d[1]} {d[2]} {d[3]}'.format(d=sizes) + ']'

    for si in range(1, len(sizes)):
        '''Recall from VL_DSIFT() that the first descriptor for scale SIZE has
        center located at XC = XMIN + 3/2 SIZE (the Y coordinate is
        similar). It is convenient to align the descriptors at different
        scales so that they have the same geometric centers. For the
        maximum size we pick XMIN = 1 and we get centers starting from
        XC = 1 + 3/2 MAX(OPTS.SIZES). For any other scale we pick XMIN so
        that XMIN + 3/2 SIZE = 1 + 3/2 MAX(OPTS.SIZES).
        In pracrice, the offset must be integer ('bounds'), so the
        alignment works properly only if all OPTS.SZES are even or odd.'''

        off = math.floor(1 + 3 / 2 * (max(sizes) - sizes[si]))

        '''smooth the image to the appropriate scale based on the size
          of the SIFT bins'''

        sigma = sizes[si] / magnif
        # here
        ims = vl_imsmooth(im, sigma)
        # check for NaN
        infinity = float("inf")
        f = {}
        d = dict()

        # extract dense SIFT features from all channels
        for k in range(1, numChannels):
            # here NaN
            f, d = dsift.dsift(ims[k, :, :], dsiftOpts, size=sizes[si], bounds=[off, off, infinity, infinity])

        '''remove low contrast descriptors
        note that for color descriptors the V component is thresholded'''

        if color_lower == 'gray' or 'opponent':
            # f is (a)x(no. of features) ndarray where a =2,3
            # contrast is 1x(no. of features) ndarray
            contrast = f[1][2, :]
        elif color_lower == 'rgb':
            m = (f[0][2, :], f[1][2, :], f[2][2, :])
            contrast = np.mean(m, axis=1)
        else:
            color_lower = 'hsv'
            contrast = f[2][2, :]

        for k in range(1, numChannels):
            d[k][:, d[k].contrast < contrastthreshold] = 0

        stack2 = (sizes[si]) * np.ones(1, (f[1][1]).shape)
        frames[si] = np.row_stack(f[1][(0, 1), :], stack2)
        descrs[si] = np.concatenate((d[:]), axis=0)


    return frames, descriptors
