# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:03:38 2015

@author: Sean Violante
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


import cv2

pract_dir=r'C:\Users\Sean Violante\Documents\Projects\practical-instance-recognition-2015a'
f=r'/data/oxbuild_lite/all_souls_000002.jpg'

im1=plt.imread(pract_dir+f)
im1a=cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
im2=ndimage.rotate(im1a,35)
im3=ndimage.interpolation.zoom(im2,0.7)
#%matplotlib qt

plt.subplot(121)
plt.imshow(im1a,plt.gray())
plt.subplot(122)
plt.imshow(im3)


import cyvlfeat.sift.dsift as dsift
import cyvlfeat.sift.sift as sift

x=dsift.dsift(im1a,verbose=True)
y=sift.sift(im1a,verbose=1)