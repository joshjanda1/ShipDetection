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
from skimage.data import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util import montage
import time

ship_path = 'F:/Machine Learning/ShipDetection/'
img_path = 'F:/Machine Learning/ShipDetection/train_v2/'
train_segment = pd.read_csv(ship_path + 'train_ship_segmentations_v2.csv')
img_ids = os.listdir(img_path)
#preprocessing
#we want to remove all images without ships in them. This will resolve class imbalances.
#https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb

imgs_w_ships = train_segment.ImageId[train_segment.EncodedPixels.isnull()==False]
imgs_w_ships = np.unique(imgs_w_ships.values) 

#thanks https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes
#ref https://www.kaggle.com/paulorzp/run-length-encode-and-decode

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

'''
most of the following code below is
used for testing, feel free to comment some out
above and bottom code will have comments to indicate tests
'''
# test to make sure functions above are working and create bounding boxes
for i in range(5):
    image = imgs_w_ships[i]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    img_0 = imread(img_path + image)
    rle_0 = train_segment[train_segment.ImageId == image]['EncodedPixels']
    mask_0 = masks_as_image(rle_0).reshape(768, 768) # reshape from 768x768x1 to 768x768
    lbl_0 = label(mask_0) 
    props = regionprops(lbl_0)
    img_1 = img_0.copy()
    print ('Image', image)
    for prop in props:
        print('Found bbox', prop.bbox)
        cv2.rectangle(img_1, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)


    ax1.imshow(img_0)
    ax1.set_title('Image')
    ax2.set_title('Mask')
    ax3.set_title('Image with derived bounding box')
    ax2.imshow(mask_0, cmap='gray')
    ax3.imshow(img_1)
    plt.show()
# end test
# convert masks of all images to bounding boxes
bbox_imgs = {}
start_time = time.time()
for i in range(len(imgs_w_ships)):
    
    img_id = imgs_w_ships[i]
    image = cv2.imread(img_path + img_id) # read image in as np array
    rle = train_segment.EncodedPixels[train_segment.ImageId == img_id] # get encoded pixels for correct img id
    mask = masks_as_image(rle).reshape(768, 768) # reshape from 768x768x1 to 768x768
    lbl = label(mask) # label the connected regions in mask array
    props = regionprops(lbl) # generate bounding boxes for labeled masks
    bboxes = []
    for prop in props: # iterate over region props
        bboxes.append(prop.bbox) # add each bounding box for each ship in image to list
    bbox_imgs['{0}'.format(img_id)] = bboxes # insert into dictionary
    
    if i % 500 == 0:
        current_time = time.time()
        percent = (i / len(imgs_w_ships))*100
        print('Images Processed: {0}/{1} - {2:.3f}%'.format(i, len(imgs_w_ships), percent))
        print('Processing Time Elapsed: {:.1f}s'.format(current_time - start_time))

bbox_imgs_df = pd.DataFrame([bbox_imgs]) # convert dict to pandas dataframe
bbox_imgs_df = bbox_imgs_df.transpose() # tranpose to get correct format
bbox_imgs_df = bbox_imgs_df.reset_index() # reset index to get image id column
bbox_imgs_df.columns = ['ImageId', 'bbox_list'] # rename columns
#bbox_imgs_df.to_csv(ship_path + 'bbox_imgs.csv')
# test to make sure dataframe is appropriately setup
for i in range(5):
    
    img_id = imgs_w_ships[i]
    image = cv2.imread(img_path + img_id)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    bboxes = bbox_imgs_df.bbox_list[bbox_imgs_df.ImageId == img_id]
    image1 = image.copy()
    bboxes = bboxes.reset_index(drop = True) # easier to access tuples in list
    for bbox in bboxes[0]:
        cv2.rectangle(image1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    ax1.imshow(image)
    ax1.set_title('Image: {0}'.format(img_id))
    ax2.set_title('Image with bounding box')
    ax2.imshow(image1)
    plt.show()
# used for checking df

sample_img_df = bbox_imgs_df.sample(100)

# thanks https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
def resize_image(img, bboxes, resize = 224):
    
    y_ = img.shape[0]
    y_scale = resize / y_
    
    x_ = img.shape[1]
    x_scale = resize / x_
    
    img = cv2.resize(img, (resize, resize))
    bboxes = bboxes.reset_index(drop = True) # easier to access tuples in list
    bboxes2 = []
    for bbox in bboxes[0]:
        x = int(np.round(bbox[0] * x_scale))
        xmax = int(np.round(bbox[2] * x_scale))
        y = int(np.round(bbox[1] * y_scale))
        ymax = int(np.round(bbox[3] * y_scale))
        bboxes2.append((x, y, xmax, ymax))
    return bboxes2, img

# test to make sure rescale worked properly
for i in range(5):
    
    img_id = imgs_w_ships[i]
    image = cv2.imread(img_path + img_id)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    bboxes = bbox_imgs_df.bbox_list[bbox_imgs_df.ImageId == img_id]
    bboxes = bboxes.reset_index(drop = True) # easier to access tuples in list
    
    res_bboxes, res_image = resize_image(image, bboxes)
    for bbox in bboxes[0]:
        cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    for bbox in res_bboxes:
        cv2.rectangle(res_image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    
    ax1.imshow(image)
    ax1.set_title('Image: {0} - 768x768'.format(img_id))
    ax2.set_title('Image: {0} - 224x224'.format(img_id))
    ax2.imshow(res_image)
    plt.show()

# end test
    

#thanks http://puzzlemusa.com/2018/04/24/resnet-in-keras/
#thanks https://arxiv.org/pdf/1512.03385.pdf
def resnet():
    #input layer
    inpt = keras.layers.Input(shape = (224, 224, 3), name = 'input')
    #1st conv layer - begin resnet
    conv1 = keras.layers.Conv2D(64, (7, 7), strides = (2, 2), padding = 'same', activation = 'relu',
                        kernel_initializer = 'he_normal', name = 'conv1', input_shape = (128, 128, 3))(inpt)
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
    output = keras.layers.Dense(1, kernel_initializer = 'he_normal',
                               activation = 'sigmoid', name = 'output')(dense2)
    
    model = keras.models.Model(inpt, output)
    
    return model

model = resnet()

optimizer = keras.optimizers.RMSprop(.0001)
model.compile(
    optimizer=optimizer, 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

history = model.fit(
    train_generator, 
    steps_per_epoch=train_steps,
    validation_data=validate_generator,
    validation_steps=validate_steps,
    epochs=epochs
)
