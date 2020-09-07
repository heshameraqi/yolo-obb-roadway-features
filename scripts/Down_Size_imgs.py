import glob
import random
import os
import numpy as np
import scipy

import torch

import random

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

'''
this script is used to down size image and save it to the disk.
It also change the labels file to the new shapes of the image. 
'''

imgs_folder = "Train"
img_shape = (468, 832)
output_folder = "../../Train_Resized"

img_paths = [imgs_folder + '/' + s for s in os.listdir(imgs_folder) if s.endswith(".jpg")]
for img_path in img_paths:

    img = np.array(Image.open(img_path))

    # load labels
    label_path = img_path.replace(".jpg", ".txt")
    labels = np.loadtxt(label_path, delimiter=' ', skiprows=1)
    if len(labels.shape) == 1:
        labels = labels.reshape(1, -1)

    h, w, _ = img.shape
    # Resize and normalize
    input_img = resize(img, (*img_shape, 3), mode='reflect')
    #plt.imshow(input_img)
    #plt.show()
    h_sized, w_sized, _ = input_img.shape

    #---------
    #  Label
    #---------

    # In cu-obb-roadway-features dataset (1K images train and test):
    #   class_id,
    #   x (centre, starts from image left limit),
    #   y (centre, starting from image upper limit),
    #   height (longest dimension),
    #   width (smallest dimension),
    #   orientation (- 0, / 45 , \ -45 , | -90 or 90)
    # In coco 2014 dataset (82K+40K images train and test): class_id,x,y,width,height (all normalized from 0 to 1)






    # Get OBB 4 vertices from[boxes, x,y,l,w,theta] that is the same order as in the labelImg_OBB tool
    p1_x = labels[:,1] + labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 + \
                         labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
    p1_y = labels[:,2] - labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 - \
                         labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

    p2_x = labels[:,1] - labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 + \
                         labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
    p2_y = labels[:,2] + labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 - \
                         labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

    p3_x = labels[:,1] - labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 - \
                         labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
    p3_y = labels[:,2] + labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 + \
                         labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

    p4_x = labels[:,1] + labels[:,3] * np.cos(np.radians(labels[:,5]     )) / 2.0 - \
                         labels[:,4] * np.cos(np.radians(90 + labels[:,5])) / 2.0
    p4_y = labels[:,2] - labels[:,3] * np.sin(np.radians(labels[:,5]     )) / 2.0 + \
                         labels[:,4] * np.sin(np.radians(90 + labels[:,5])) / 2.0

    # Adjust vertices for added padding
    p1_x = (p1_x / w) * w_sized
    p2_x = (p2_x / w) * w_sized
    p3_x = (p3_x / w) * w_sized
    p4_x = (p4_x / w) * w_sized
    p1_y = (p1_y / h) * h_sized
    p2_y = (p2_y / h) * h_sized
    p3_y = (p3_y / h) * h_sized
    p4_y = (p4_y / h) * h_sized

    # Normalize origin for yolo GT to be from 0 to 1
    labels[:, 1] = (p1_x+p2_x+p3_x+p4_x) / 4
    labels[:, 2] = (p1_y+p2_y+p3_y+p4_y) / 4

    val1 = np.sqrt(((p2_x - p1_x) ** 2) + ((p2_y - p1_y) ** 2))
    val2 = np.sqrt(((p3_x - p2_x) ** 2) + ((p3_y - p2_y) ** 2))
    labels[:, 3] = np.max([val1, val2], axis=0)  # width
    labels[:, 4] = np.min([val1, val2], axis=0)  # length

    im = Image.fromarray(np.uint8(input_img*255))
    im.save(output_folder+ img_path.replace(imgs_folder, ''))

    with open(output_folder + img_path.replace(".jpg", ".txt").replace(imgs_folder, ''), 'w') as file:
        file.write("YOLO_OBB" + "\n")
        for lab in labels:
            file.write("%d %.5f %.5f %.5f %.5f %.5f" % (lab[0], lab[1], lab[2], lab[3], lab[4], lab[5]) + "\n")

    # save the image resized
