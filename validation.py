# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:18:57 2020

@author: Josh
"""

import pandas as pd
import numpy as np
from PIL import Image
import time
from shapely.geometry import box
from shapely.ops import unary_union
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ship_path = 'F:/Machine Learning/ShipDetection/'
image_path = 'F:/Machine Learning/ShipDetection/train_v2/'
validation_path = 'F:/Machine Learning/ShipDetection/validation/'
resize_path = 'F:/Machine Learning/ShipDetection/train_v2_resized_v2/'
'''
annotations_final = pd.read_csv(ship_path + 'annotate.txt', header = None)
annotations_final.columns = ['file_path', 'x1', 'y1', 'x2', 'y2', 'class_name']
val_annotations = pd.read_csv(ship_path + 'val_annotations.csv', header = None)
val_annotations.columns = ['x1', 'y1', 'x2', 'y2', 'class_name', 'img_id']
prediction_annotations = pd.read_csv(ship_path + 'prediction_annotations.csv', header = None)
prediction_annotations.columns = ['x1', 'y1', 'x2', 'y2', 'img_id']
'''

def generate_validation_data(annotations, image_path, validation_path, split_on = '/', seed = 27, size = .25, verbose = True):
    '''
    Parameters
    ----------
    annotations : dataframe
        annotations dataframe consisting of bounding boxes and file path
    image_path : str
        Directory to original images
    validation_path : str
        path to save images in validation set
    split_on : str
        Character to split file_path string on. Default is '/'
    seed : int
        Random seed to initiate. Default is 27
    size : float
        Size of validation set. Default is 25% of training data
    verbose: boolean
        Should function print progress? Default is True.

    Returns
    -------
    val_annotations
        annotations file for validation dataset

    '''
    
    image_ids = [img_id.split(split_on)[1] for img_id in annotations["file_path"]] # grab img id from file path
    unique_image_ids = np.unique(image_ids) # only keep unique image ids
    
    np.random.seed(seed)
    val_ids = np.random.choice(unique_image_ids, size = int(size*len(unique_image_ids)), replace = False) # sample ids for validation data
    annotations['img_id'] = image_ids
    val_annotations = annotations[annotations['img_id'].isin(val_ids)] # create dataframe containing only images in val_ids
    val_annotations.drop('file_path', axis = 1, inplace = True) # drop file path as we only need image id
    
    start = time.time() # get start time
    tot_img = len(val_ids)
    
    for i, img_id in enumerate(val_ids):
        
        im = Image.open(image_path + img_id)
        im.save(validation_path + img_id)
        
        if i % 500 == 0 and verbose:
            curr = time.time() 
            mins = (curr - start) / 60
            print('Images Processed: {0}/{1} - Time Elasped: {2:.2f}m'.format(i, tot_img, mins))
    
    return val_annotations

def calculate_mean_iou(true_annotations, predicted_annotations):
    
    '''
    Calculate the mean Intersection over Union (IoU) for all images
    
    Parameters
    ----------
    true_annotations : dataframe
        Dataframe consisting of true annotations for image data
        x1, y1, x2, y2 coordinates and image id
    predicted_annotations : dataframe
        Dataframe consisting of predicted annotations for image data
        x1, y2, x2, y2 coordinates and image id
    
    Returns
    -------
    float
        in [0, 1]
    '''
    
    ious = []
    
    for img_id in np.unique(true_annotations['img_id']):
        
        true = true_annotations[true_annotations['img_id'] == img_id] # get true bounding boxes for current image
        predicted = predicted_annotations[predicted_annotations['img_id'] == img_id] # get predicted bounding boxes for current image
        
        #create list of box objects for each bounding box in true annotations
        true_bboxes = [box(x1, y1, x2, y2) for i, (x1, y1, x2, y2, class_name, img_id) in true.iterrows()]
        
        #case when model predicts no instances of ships in an image so predicted is empty
        if len(predicted) == 0:
            iou = 0 # if nothing predicted then intersection over union is zero
            ious.append(iou)
            continue
        #create list of box objects for each bounding box in predicted annotations
        predicted_bboxes = [box(x1, y1, x2, y2) for i, (x1, y1, x2, y2, img_id) in predicted.iterrows()]
        
        #generate union of true bounding boxes
        true_union = unary_union(true_bboxes)
        #generate union of predicted bounding boxes
        predicted_union = unary_union(predicted_bboxes)
        #calculate intersection of true union and predicted union
        intersection = true_union.intersection(predicted_union)
        #calculate intersection over union : https://en.wikipedia.org/wiki/Jaccard_index
        iou = intersection.area / true_union.area
        ious.append(iou)
    #compute mean intersection over union
    return np.mean(ious)

def show_images(true_annotations, predicted_annotations, resize_path, n = 5, seed = 27):
    
    '''
    Display images with true and predicted bounding box overlayed
    
    Parameters
    ----------
    true_annotations : dataframe
        Dataframe consisting of true annotations for image data
        x1, y1, x2, y2 coordinates and image id
    predicted_annotations : dataframe
        Dataframe consisting of predicted annotations for image data
        x1, y2, x2, y2 coordinates and image id
    resize_path : str
        String consisting of path to resized image data
    n : int
        Number of images to randomly sample for display
        Default: 5
    seed : int
        Seed to initialize to random state to maintain same images between function calls
        Default : 27
    
    Returns
    -------
    None 
    '''
    #useable images are all images in predicted annotations (val set)
    useable_images = predicted_annotations['img_id']
    #sample n images to display
    useable_images = useable_images.sample(n, random_state = seed)
    
    for img_id in useable_images.values:
        
        img = cv2.imread(resize_path + str(img_id))
        true_details = true_annotations[true_annotations['img_id'] == img_id]
        predicted_details = predicted_annotations[predicted_annotations['img_id'] == img_id]
        
        for i, (x1, y1, x2, y2, class_name, img_id) in true_details.iterrows():
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color = (255, 0, 0), thickness = 2)
        
        for i, (x1, y1, x2, y2, img_id) in predicted_details.iterrows():
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2)
        
        fig, ax = plt.subplots(1)
        true_patch = mpatches.Patch(color = 'red', label = 'True Bounding Box')
        pred_patch = mpatches.Patch(color = 'blue', label = 'Predicted Bounding Box')
        ax.imshow(img)
        ax.set_title('Image ID: {0}'.format(img_id))
        ax.legend(handles = [true_patch, pred_patch])
        plt.show()
        
        
        
