#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:16:42 2017

@author: jingang
"""

from keras.models import model_from_json
import cv2
import numpy as np
import sys

img_path = sys.argv[1]
# load json and create model
#def predict(img_path):
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
model=loaded_model
best_threshold=np.load('./threshold.npy')/4
img = cv2.imread(img_path)
    
    
    
img = cv2.resize(img,(100,100))
img = img.transpose((2,0,1))
img = img.astype('float32')
img /= 255
img = np.expand_dims(img,axis=0)
pred = model.predict(img)
y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])
classes = ['desert','mountains','sea','sunset','trees']
#print [classes[i] for i in range(5) if y_pred[i]==1 ]  #extracting actual class name
print [[pred[0,i],classes[i]] for i in range(5)  ]



    

