based on https://github.com/bonlime/keras-deeplab-v3-plus, pre-liminary work from nigel and many other example codes on smaller, individual functions 
such as padding, super impose images, add label etc
used a state-of-the-art model deeplab v3+, extracted a small fraction of its features to identify 20 different obj, inc. dog & cat
dataset from https://www.kaggle.com/c/dogs-vs-cats/data

produce error report,
compare performance with first network
refined the output to add labels 

accuracy 98.1%

run predictPic.py to make a prediction on an image
put own images under imgs folder
run findAccuracy to find % accuracy of network

extract_weights, load_weights and model are used to build the DeepLabv3+ model
I wrote padding.py to just create a nice way to pad an image to the required dimension
util.py contains some methods that I used for plotting and overlaying transparent images
