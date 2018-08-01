# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:06:08 2018

@author: minji
"""

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import cv2

def pad_img(im_pth, desired_size):

    im = plt.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0])) 

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=color)
    return img