# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:17:59 2020

@author: Josh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import cv2
import random

ship_path = 'F:/Machine Learning/ShipDetection/'
img_path = 'F:/Machine Learning/ShipDetection/train_v2/'
train_segment = pd.read_csv(ship_path + 'train_ship_segmentations_v2.csv').set_index('ImageId')
img_ids = os.listdir(img_path)
#preprocessing
#we want to remove all images without ships in them. This will resolve class imbalances.
#https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb

train_segment = train_segment.dropna()
uni = np.unique(train_segment.index, return_index=True)[1]
train_segment = train_segment.iloc[uni]

sample_segment = train_segment.sample(n = 100, random_state = 27)
img_ids = [name for name in img_ids if name in sample_segment.index] # keep all img ids that are in segment df
imgs = []
for i, img_id in enumerate(img_ids):
    img = np.asarray(Image.open(img_path + img_id))
    imgs.append(img)
    
imgs = np.array(imgs)

Xtrain = imgs
#thanks http://puzzlemusa.com/2018/04/24/resnet-in-keras/
#thanks https://arxiv.org/pdf/1512.03385.pdf
def resnet():
    #input layer
    inpt = keras.layers.Input(shape = (768, 768, 3), name = 'input')
    #1st conv layer - begin resnet
    conv1 = keras.layers.Conv2D(64, (7, 7), strides = (2, 2), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv1', input_shape = Xtrain.shape[1:])(inpt)
    norm1 = keras.layers.BatchNormalization(axis = 3, name = 'normal1')(conv1)
    relu1 = keras.layers.Activation('relu', name = 'relu1')(norm1)
    mpool1 = keras.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'pool1')(relu1)
    #2nd conv layer - no add
    conv2 = keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv2')(mpool1)
    norm2 = keras.layers.BatchNormalization(axis = 3, name = 'normal2')(conv2)
    relu2 = keras.layers.Activation('relu', name = 'relu2')(norm2)
    #3rd conv layer - add
    conv3 = keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv3')(relu2)
    add1 = keras.layers.add([mpool1, conv3], name = 'add1')
    norm3 = keras.layers.BatchNormalization(axis = 3, name = 'normal3')(add1)
    relu3 = keras.layers.Activation('relu', name = 'relu3')(norm3)
    #4th conv layer - no add
    conv4 = keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv4')(relu3)
    norm4 = keras.layers.BatchNormalization(axis = 3, name = 'normal4')(conv4)
    relu4 = keras.layers.Activation('relu', name = 'relu4')(norm4)
    #5th conv layer - add
    conv5 = keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv5')(relu4)
    add2 = keras.layers.add([add1, conv5], name = 'add2')
    norm5 = keras.layers.BatchNormalization(axis = 3, name = 'normal5')(add2)
    relu5 = keras.layers.Activation('relu', name = 'relu5')(norm5)
    #6th conv layer - no add
    conv6 = keras.layers.Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv6')(relu5)
    norm6 = keras.layers.BatchNormalization(axis = 3, name = 'normal6')(conv6)
    relu6 = keras.layers.Activation('relu', name = 'relu6')(norm6)
    #7th conv layer - add
    conv7 = keras.layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv7')(relu6)
    conv7_2 = keras.layers.Conv2D(128, (1, 1), strides = (2, 2), padding = 'valid', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv7_2')(add2)
    add3 = keras.layers.add([conv7, conv7_2], name = 'add3')
    norm7 = keras.layers.BatchNormalization(axis = 3, name = 'normal7')(add3)
    relu7 = keras.layers.Activation('relu', name = 'relu7')(norm7)
    #8th conv layer - no add
    conv8 = keras.layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv8')(relu7)
    norm8 = keras.layers.BatchNormalization(axis = 3, name = 'normal8')(conv8)
    relu8 = keras.layers.Activation('relu', name = 'relu8')(norm8)
    #9th conv layer - add
    conv9 = keras.layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv9')(relu8)
    add4 = keras.layers.add([add3, conv9], name = 'add4')
    norm9 = keras.layers.BatchNormalization(axis = 3, name = 'normal9')(add4)
    relu9 = keras.layers.Activation('relu', name = 'relu9')(norm9)
    #9th layer continued - jump between dotted lines in pdf architecture
    conv9 = keras.layers.Conv2D(256, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv9_2')(relu9)
    norm9 = keras.layers.BatchNormalization(axis = 3, name = 'normal9_2')(conv9)
    relu9 = keras.layers.Activation('relu', name = 'relu9_2')(norm9)
    #10th conv layer - add
    conv10 = keras.layers.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv10')(relu9)
    conv10_2 = keras.layers.Conv2D(256, (2, 2), strides = (2, 2), padding = 'valid', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv10_2')(add4)
    add5 = keras.layers.add([conv10, conv10_2], name = 'add5')
    norm10 = keras.layers.BatchNormalization(axis = 3, name = 'normal10')(add5)
    relu10 = keras.layers.Activation('relu', name = 'relu10')(norm10)
    #11th conv layer - no add
    conv11 = keras.layers.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv11')(relu10)
    norm11 = keras.layers.BatchNormalization(axis = 3, name = 'normal11')(conv11)
    relu11 = keras.layers.Activation('relu', name = 'relu11')(norm11)
    #12th conv layer - add
    conv12 = keras.layers.Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv12')(relu11)
    add6 = keras.layers.add([add5, conv12], name = 'add6')
    norm12 = keras.layers.BatchNormalization(axis = 3, name = 'normal12')(add6)
    relu12 = keras.layers.Activation('relu', name = 'relu12')(norm12)
    #13th conv layer - no add - increase to 512 filters
    conv13 = keras.layers.Conv2D(512, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv13')(relu12)
    norm13 = keras.layers.BatchNormalization(axis = 3, name = 'normal13')(conv13)
    relu13 = keras.layers.Activation('relu', name = 'relu13')(norm13)
    #14th conv layer - add
    conv14 = keras.layers.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv14')(relu13)
    conv14_2 = keras.layers.Conv2D(512, (1, 1), strides = (2, 2), padding = 'valid', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv14_2')(add6)
    add7 = keras.layers.add([conv14, conv14_2], name = 'add7')
    norm14 = keras.layers.BatchNormalization(axis = 3, name = 'normal14')(add7)
    relu14 = keras.layers.Activation('relu', name = 'relu14')(norm14)
    #15th conv layer - no add
    conv15 = keras.layers.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv15')(relu14)
    norm15 = keras.layers.BatchNormalization(axis = 3, name = 'normal15')(conv15)
    relu15 = keras.layers.Activation('relu', name = 'relu15')(norm15)
    #16th conv layer - add - final layer
    conv16 = keras.layers.Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv16')(relu15)
    add8 = keras.layers.add([add7, conv16], name = 'add8')
    norm16 = keras.layers.BatchNormalization(axis = 3, name = 'normal16')(add8)
    relu16 = keras.layers.Activation('relu', name = 'relu16')(norm16)
    #avg pooling layer
    apool1 = keras.layers.AveragePooling2D(pool_size = (1, 1), strides = (1, 1))(relu16)
    flatten = keras.layers.Flatten()(apool1)
    dense1 = keras.layers.Dense(4096, kernel_initializer = 'he_normal',
                               activation = 'relu', name = 'fc1')(flatten)
    dense2 = keras.layers.Dense(4096, kernel_initializer = 'he_normal',
                               activation = 'relu', name = 'fc2')(dense1)
    output = keras.layers.Dense(10, kernel_initializer = 'he_normal',
                               activation = 'softmax', name = 'output')(dense2)
    
    model = keras.models.Model(inpt, output)
    
    return model

model = resnet()




