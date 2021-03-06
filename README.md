# Information

This is the repository for work/models on the Airbus Ship Detection Challenge through [Kaggle](https://www.kaggle.com/c/airbus-ship-detection/overview).

Prepocessing Steps:

1. Resize training images to 256x256, save to seperate resized folder
2. Create bounding boxes by utilizing masks given in **train_submissions_v2.csv** file and pushing through `create_bboxes` function
3. Using bounding box dataframe, use `label_df` function to resize bounding boxes and create one observation per ship per images
4. Using labeled bounding box dataframe, run through `reformat_labeled_df` function to format to required output for training

Training Steps:

For training, I am using this [Keras implementation of Faster RCNN](https://github.com/kbardool/keras-frcnn)

Note that code will be changed in the implementation to work with newest versions of TensorFlow (2.1.0) and Keras (2.3.1).

Some added changes are that I have:

- Slightly edited test_frcnn.py file for sending predicted annotations to a csv file for use on validation set
- Added the ability to log losses during training through tensorboard

Requirements:

- Tensorflow GPU 2.1.0
- Keras 2.3.1
- Pillow 7.0.0
- Opencv-Python 4.1.2.30
- Numpy 1.18.1
- Pandas 1.0.3
- Scikit-Image 0.15.0
- Matplotlib 3.1.3
- Shapely

1. Clone repository
2. Either move repository to main directory that contains training/testing image directories or move training/testing image directories to model repository
3. In any model files containing **K.image_dim_ordering()**, replace with **K.common.image_dim_ordering()**
4. In `FixedBatchNormalization.py`, in any `self.add_weight()` calls change shape to shape = shape
5. In config.py, change `self.im_size` to 256
6. Navigate to model folder and run python train_frcnn.py -o simple -p annotate.txt --num_epochs number_of_epochs --input_weight_path ./model_frcnn_256.hdf5 --output_weight_path ./model_frcnn_256.hdf5

Examples:

![](Images/samplebbox1.png) ![](Images/samplebbox2.png)
![](Images/predicted1.png)  ![](Images/predicted3.png)

Known Issues:

- When training, I am getting an error "Exception: 'a' cannot be empty unless no samples are taken".
	- This may be due to no negative samples (no background only images)
	- I think it can be safely ignored, but possibly take a look at https://github.com/kbardool/keras-frcnn/issues/21
