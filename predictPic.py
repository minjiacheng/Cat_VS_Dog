from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3
import util
import yaml
from padding import pad_img
deeplab_model = Deeplabv3()
f = open("pascal_voc.yaml", 'r')
DATA = yaml.load(f)
from time import time

### To make new predictions, run codes below this line
start = time()
im_pth = "imgs/dog.jpg" #put the file path here
desired_size = 512
img = pad_img(im_pth, desired_size) #read input img and pre-process to required shape
#preprocess and plot segmentation map
w, h, _ = img.shape
ratio = 512. / np.max([w,h])
resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
resized = resized / 127.5 - 1.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
res = deeplab_model.predict(np.expand_dims(resized2,0))
labels = np.argmax(res.squeeze(),-1)

#create transparent overlay
color_mask = util.prediction_to_color(labels, DATA["label_remap"], DATA["color_map"])
input_img, seg_map, seg_overlay = util.transparency(img, color_mask)

#create labels for overlay
label_list = np.unique(labels) #list of unique entries from seg map
#label_list = [0,8,12,18] #test data
objects = ['void', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
           'Chair', 'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person',
           'Pottedplant', 'Sheep', 'Sofa', 'Train', 'TVmonitor']
colours = [(0,0,0),(0,128,0),(128,128,0),(0,0,128),(192,128,0),(128,0,128),
           (0,128,128),(128,128,128),(64,0,128),(192,0,0),(64,128,0),
           (192,128,0),(64,0,0),(192,0,128),(64,128,128),(192,128,128),
           (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
#copy references from yaml
label_index = np.full((240,240,3), 255, np.uint8)
#blank figure
counter = 20
for entry in label_list:
    label_index = cv2.rectangle(label_index, (30,counter), (70,counter+40), colours[entry], -1)
    #draw squares
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(label_index,objects[entry],(100,counter+30), font, 1,colours[entry],2,cv2.LINE_AA)
    #add labels
    counter +=60  

#plot all images
plt.figure()
plt.imshow(input_img) 
plt.axis('off')
plt.title("input image") 
f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(seg_map) 
plt.axis('off')
plt.title("segmentation map") 
f.add_subplot(1,2, 2)
plt.imshow(label_index) 
plt.axis('off')
plt.tight_layout()
plt.show(block=True)
#plot seg_map and label side by side
plt.figure()
plt.imshow(seg_overlay) 
plt.axis('off')
plt.title("segmentation overlay") 

#count number of cat and dog pixels
cat_count = np.count_nonzero(labels == 8)
dog_count = np.count_nonzero(labels == 12)
#if there's more cat pixels, predict it's a cat, vice versa
if cat_count > dog_count:
    print("I predict this is a cat.")
elif dog_count > cat_count:
    print("I predict this is a dog.")
else:
    print("Ops looks like I failed to spot a cat or dog.")
    
end = time()
print("time taken for the prediction is: ", end-start, "s")