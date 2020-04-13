import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os

ship_path = 'F:/Machine Learning/ShipDetection/'

def rescale_imgs(directory, to_directory, size):
    start = time.time()
    # code to get images with ships
    # only want unique images with ships
    train_segment = pd.read_csv(directory + 'train_ship_segmentations_v2.csv')
    imgs_w_ships = train_segment.ImageId[train_segment.EncodedPixels.isnull()==False]
    imgs_w_ships = np.unique(imgs_w_ships.values)
    tot_img = len(imgs_w_ships)
    # end code
    for i, img in enumerate(imgs_w_ships):
        im = Image.open(directory + 'train_v2/' + img)
        im_res = im.resize(size, Image.ANTIALIAS)
        im_res.save(to_directory + img)
        
        if i % 500 == 0:
            curr = time.time() 
            mins = (curr - start) / 60
            print('Images Processed: {0}/{1} - Time Elasped: {2:.2f}m'.format(i, tot_img, mins))

rescale_imgs(ship_path, ship_path + 'train_v2_resized/', (224, 224))
 