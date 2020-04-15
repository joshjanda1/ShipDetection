import numpy as np
import pandas as pd
from PIL import Image
import time

ship_path = 'F:/Machine Learning/ShipDetection/'

def rescale_imgs(directory, to_directory, size = (256, 256)):
    '''
    Parameters
    ----------
    directory : str
        Main directory of images. in this case, use the ship_path defined at the top of the file.
    to_directory : str
        Directory to save resized images.
    size : tuple
        Size of the image. Default is (256, 256)

    Returns
    -------
    None.

    '''
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