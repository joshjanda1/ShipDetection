# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:17:59 2020

@author: Josh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops
import time
import ast


ship_path = 'F:/Machine Learning/ShipDetection/'
img_path = 'F:/Machine Learning/ShipDetection/train_v2/'
resize_path = 'F:/Machine Learning/ShipDetection/train_v2_resized/'
resize_path2 = 'F:/Machine Learning/ShipDetection/train_v2_resized_v2/'
train_segment = pd.read_csv(ship_path + 'train_ship_segmentations_v2.csv')


imgs_w_ships = train_segment.ImageId[train_segment.EncodedPixels.isnull()==False]
imgs_w_ships = np.unique(imgs_w_ships.values) 

'''
The code below is used for testing purposes.
If you have saved the dataframes outputted from functions
create_bbox, resize_and_relabel_bbox, and reformat_label_df
then use the code to load the csvs to avoid reprocessing.

If processed correctly, you should not need to rerun these utilities.


bbox_df = pd.read_csv(ship_path + 'bbox_imgs.csv')
resized_bbox_df = pd.read_csv(ship_path + 'labeled_ships.csv')
annotations_final = pd.read_csv(ship_path + 'annotate.txt', header = False)
annotations_final.columns = ['file_path', 'x1', 'y1', 'x2', 'y2', 'class_name']
'''




#thanks https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes
#ref https://www.kaggle.com/paulorzp/run-length-encode-and-decode

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

def img_w_bbox(bbox_df, img_path, resize_path, resize = True, size = 256, num_imgs = 5, seed = 27):
    
    '''
    function used to test if bounding boxes align with ship in images
    
    args:
        bbox_df - dataframe consisting of ImageId and list of tuples of bouning boxes (output from create_bboxes function)
        img_path - path to original images
        resize_path - path to resized images
        resize - Default: True - overlay resized images with resized bounding boxes? Set to False to use original image and original bbox
        num_imgs - number of images to view
        seed - seed sample to obtain same pictures
    output:
        side-by-side plot of original (or resized) image and original (or resized) image overlayed with appropriate bounding box(es)
        
    '''
    
    sample = bbox_df.sample(num_imgs, random_state = seed)
    for i, (img_id, bboxes) in sample.iterrows():
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5)) # initiate plots
        if resize:
            img = cv2.imread(resize_path + img_id) # get resized image if resize is True
        else:
            img = cv2.imread(img_path + img_id) # get original image if resize is False
        img_overlay = img.copy() # copy image to overlay
        bboxes = ast.literal_eval(bboxes) # needed after reloading bbox_df from csv file as bbox lists are treated as strings
        for bbox in bboxes:
            if resize:
                x, y, xmax, ymax = resize_bbox(bbox)
                cv2.rectangle(img_overlay, (y, x), (ymax, xmax), (255, 0, 0), 2)
            else:
                x, y, xmax, ymax = bbox
                cv2.rectangle(img_overlay, (y, x), (ymax, xmax), (255, 0, 0), 2)
        ax1.imshow(img)
        ax2.imshow(img_overlay)
        if resize: # plot titles if resized images..
            ax1.set_title('Image ID: {0}\nImage Size: {1}x{2}'.format(img_id, size, size))
            ax2.set_title('Image ID: {0}\nWith Resized Bounding Box(es)'.format(img_id))
        else: # plot titles if original images..
            ax1.set_title('Image ID: {0}\nImage Size: 768x768'.format(img_id))
            ax2.set_title('Image ID: {0}\nWith Original Bounding Box(es)'.format(img_id))
        plt.show()
        
def create_bboxes(imgs_w_ships, segment_df,  ship_path, img_size = 768, to_csv = False):
    
    '''
    function is used to convert encoded pixel masks to bounding boxes
    for localizing object detection
    args:
        imgs_w_ships - list of all images containing ships (may be more than one ship per image)
        segment_df - dataframe containing ImageIds and corresponding encoded pixels of masks
        ship_path - path to main directory for model
        img_size - original image size
        to_csv - Default: False - set to true to export created dictionary to csv file of imageid and corresponding bboxes
    returns:
        pandas dataframe consisting of columns 'ImageId' and 'bbox_list'
        where ImageId is the image id and bbox_list is a list of tuples containing bounding boxes in each image
    '''
    
    bbox_imgs = {} # create empty dictionary to store images
    start = time.time() # used to compute elasped time
    
    for i in range(len(imgs_w_ships)):
        
        img_id = imgs_w_ships[i]
        rle = segment_df.EncodedPixels[segment_df.ImageId == img_id] # get encoded pixels for correct img id
        mask = masks_as_image(rle).reshape(768, 768) # reshape from 768x768x1 to 768x768
        lbl = label(mask) # label the connected regions in mask array
        props = regionprops(lbl) # generate bounding box regions for labeled masks
        bboxes = []
        for prop in props: # iterate over region props 
            bboxes.append(prop.bbox) # add each bounding box for each ship in image to list
        bbox_imgs['{0}'.format(img_id)] = bboxes # insert into dictionary
        
        if i % 500 == 0:
            current = time.time()
            percent = (i / len(imgs_w_ships))*100
            tot_mins = (current - start) / 60
            print('Images Processed: {0}/{1} - {2:.3f}% - Time Elasped: {3:.2f}m'.format(i, len(imgs_w_ships), percent, tot_mins))
            
    bbox_imgs_df = pd.DataFrame([bbox_imgs]) # convert dict to pandas dataframe
    bbox_imgs_df = bbox_imgs_df.transpose() # transpose to get images x 2 format
    bbox_imgs_df = bbox_imgs_df.reset_index() # reset index to get ImageId column
    bbox_imgs_df.columns = ['ImageId', 'bbox_list']
    
    if to_csv:
        bbox_imgs_df.to_csv(ship_path + 'bbox_imgs.csv', index = False)
    
    return bbox_imgs_df
        

def resize_bbox(bbox, size = 768, target = 256):
    
    '''
    thanks https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
    function used to resize bbox to appropriate target image size
    
    args:
        bbox - tuple consisting of x, y, xmax, ymax for bounding box
        size - original image size (square)
        target - target image size (square)
    returns:
        coordinates of scaled bounding box
    '''
    
    scale = target / size # scale for resize
    
    x = int(np.round(bbox[0] * scale))
    xmax = int(np.round(bbox[2] * scale))
    y = int(np.round(bbox[1] * scale))
    ymax = int(np.round(bbox[3] * scale))
    #perform rescaling of each point
    
    return x, y, xmax, ymax

def resize_and_relabel_bbox_df(bbox_df, ship_path, size = 768, target = 256, to_csv = False):
    '''
    now need to transform bbox df to be a df with each row containing an
    imageid, x, y, xmax, ymax, and a label (ship for all images)
    so there may be multiple rows of identical imageids if more than one ship in image
    
    this function performs the above description
    args:
        bbox_df - dataframe consisting of format ImageId, bbox where bbox is the list bboxes in each ImageId
        size - original size of image
        target - target image size
        ship_path - path to export csv to
        to_csv - default: False - set to true to export csv to ship_path root directory
    returns: 
        labeled dataframe of format imageid, x, y, xmax, ymax, and a label (ship for all images)
    '''
    labeled_df = pd.DataFrame([], columns = ['ImageId', 'x', 'y', 'xmax', 'ymax', 'label'])
    item_label = 'Ship' # used to label each bbox.. which is ship for all.
    start_time = time.time() # used to calculate total elapsed time
    for i, (img_id, bboxes) in bbox_df.iterrows():
        bboxes = ast.literal_eval(bboxes) # needed after reloading bbox_df from csv file as bbox lists are treated as strings
        if len(bboxes) > 1: # covers case with more than one ship in image
            for bbox in bboxes:
                # resize bboxes
                x, y, xmax, ymax = resize_bbox(bbox, size, target)
                labeled_df = labeled_df.append(pd.DataFrame([[img_id, x, y, xmax, ymax, item_label]],
                                                            columns = ['ImageId', 'x', 'y', 'xmax', 'ymax', 'label']))
        else: # covers case with only one ship in image
            for bbox in bboxes:
                # resize bboxes
                x, y, xmax, ymax = resize_bbox(bbox, size, target)
                labeled_df = labeled_df.append(pd.DataFrame([[img_id, x, y, xmax, ymax, item_label]],
                                                            columns = ['ImageId', 'x', 'y', 'xmax', 'ymax', 'label']))
        if i % 500 == 0:
            current_time = time.time()
            percent = (i / len(imgs_w_ships))*100
            print('Images Processed: {0}/{1} - {2:.3f}%'.format(i, len(bbox_df), percent))
            print('Processing Time Elapsed: {:.1f}s'.format(current_time - start_time))
    if to_csv:
        labeled_df.to_csv(ship_path + 'labeled_ships.csv', index = False)
    return labeled_df

def reformat_label_df(labeled_df, img_path, ship_path, to_csv = False):
    '''
    use to reformat labeled df to specific format required
    for pushing through model training algorithm
    
    args:
        labeled_df - pandas dataframe of format ImageId, x, y, xmax, ymax, label
        img_path - path to images
        ship_path - path to export csv to
        to_csv - default: False. Set to true to export csv to ship_path root directory
    returns:
        reformatted dataframe of format required for training
    '''
    labeled_df_temp = labeled_df.copy()
    
    x = labeled_df.x
    y = labeled_df.y
    xmax = labeled_df.xmax
    ymax = labeled_df.ymax
    labeled_df_temp.x = y
    labeled_df_temp.y = x
    labeled_df_temp.xmax = ymax
    labeled_df_temp.ymax = xmax
    
    labeled_df_temp['file_path'] = img_path + labeled_df['ImageId']
    labeled_df_temp.drop('ImageId', axis=1, inplace=True)
    cols = labeled_df_temp.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    
    labeled_df_temp = labeled_df_temp[cols]
    labeled_df_temp.columns = ['file_path', 'x1', 'y1', 'x2', 'y2', 'class_name']
    if to_csv:
        labeled_df_temp.to_csv(ship_path + 'annotate.txt', index = False, header = False)
    return labeled_df_temp
    