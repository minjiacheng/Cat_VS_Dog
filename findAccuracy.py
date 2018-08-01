# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:26:47 2018

@author: minji
"""
from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3
import os
from padding import pad_img
deeplab_model = Deeplabv3()

#give % accuracy of the network and return all false predictions
directory = './imgs'
Ycount = 0
Ncount = 0
desired_size = 512
for filename in os.listdir(directory):
    if filename.startswith("dog"): 
        label = "dog"
    elif filename.startswith("cat"): 
        label = "cat"
    img = pad_img(directory+"/"+filename, desired_size)
    w, h, _ = img.shape
    ratio = 512. / np.max([w,h])
    resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
    resized = resized / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
    res = deeplab_model.predict(np.expand_dims(resized2,0))
    labels = np.argmax(res.squeeze(),-1)
    
    #count number of cat and dog pixels
    cat_count = np.count_nonzero(labels == 8)
    dog_count = np.count_nonzero(labels == 12)
    #if there's more cat pixels, predict it's a cat, vice versa
    if cat_count > dog_count:
        prediction = "cat"
    elif dog_count > cat_count:
        prediction = "dog"
    else:
        prediction = "unknown"
    if label == prediction:
        Ycount += 1
    else:
        Ncount += 1
accuracy = (Ycount / (Ycount + Ncount))
print("accuracy is: ", accuracy)




    