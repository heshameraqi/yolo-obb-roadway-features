import glob
import random
import os
import numpy as np

import torch

import random

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

from imgaug import augmenters as iaa

import sys


# augmentations
class HorizinatFlib(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            # mirror the image
            img = img[:,::-1,:] # the secounf part is reverse casting
            # mirro the bbox
            bbox[:, 1] = img.shape[1] - bbox[:, 1] # x = width - x
            bbox[:, 5] *= -1 # angel = - angel
            # switch classes of the specific side classes
            cls = bbox[:, 0].copy()
            bbox[cls == 2, 0] = 3  # right barriar -> left
            bbox[cls == 3, 0] = 2  # left barriar -> right
            bbox[cls == 5, 0] = 6  # right curb  -> left
            bbox[cls == 6, 0] = 5  # left curb -> right


        return img, bbox

class Random_rotate(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img, labels, angel_min=-5, angel_max=5):
        if random.random() < self.p:
            #get the rotation angel
            angel = np.random.uniform(angel_min, angel_max)

            # rotate Image with saving the dims
            img_new = np.array(transforms.functional.rotate(Image.fromarray(np.uint8(img)), angel, expand=True), dtype=np.float32)

            # dims of the img before rotation
            w, h = img.shape[1], img.shape[0]
            cx, cy = w//2, h//2

            # get the new center after rotation
            w_new, h_new = img_new.shape[1], img_new.shape[0]
            cx_new, cy_new = w_new // 2, h_new // 2

            # rotate the center of the boxes with the new image
            r = np.sqrt((labels[:,1]-cx)**2 + (labels[:,2]-cy)**2)
            theta = np.arctan2((labels[:,2]-cy), (labels[:,1]-cx)) * (180 / np.pi) - angel
            labels[:, 1], labels[:, 2] = cx_new + r*np.cos(np.radians(theta)), cy_new + r*np.sin(np.radians(theta))

            # the hight and the width is the same as before

            # the rotation of the angel is the same as the ration of the image
            labels[:, 5] += angel

            img = img_new

        return img, labels

class Random_translate(object):
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img, labels):
        if random.random() < self.p:
            pass


        return img, labels



class Image_augmentation(object):

    def __init__(self):
        return

    def __call__(self, img):
        return self.create_augmenter()(image = img)

    def create_augmenter(self):
        st = lambda aug: iaa.Sometimes(0.4, aug)  # 40% of images to be augmented
        oc = lambda aug: iaa.Sometimes(0.3, aug)  # 30% of images to be augmented
        rl = lambda aug: iaa.Sometimes(0.09, aug)  # 9% of images to be augmented
        seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.7), per_channel=0.5)), # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.05))), # randomly remove up to X% of the pixels
        #oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2))), # randomly remove up to X% of the pixels
        oc(iaa.Add((-40/255, 40/255))),  # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((1.0, 1.5))), # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.75, 1.5))),  # improve or worsen the contrast
        # rl(iaa.Grayscale((0.0, 1))), # put grayscale
        ], random_order=True) # order is shuffled each time with equal probability
        return seq





class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, label_files, data_folder, img_size=416, val = False):
        with open(label_files,"r") as file:
            self.label_files = [data_folder + "/" + s.replace("\n","") for s in file.readlines()]
        self.img_files = [s.replace(".txt", ".jpg") for s in self.label_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50 # TODO: should be reduced?
        self.acess_numb = 0
        self.val = val

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        # augmentor objects
        img_augmentor = Image_augmentation()
        rand_flib = HorizinatFlib()
        rand_rotate = Random_rotate()

        # load image
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        # load labels
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        labels = np.loadtxt(label_path, delimiter=' ', skiprows=1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        # delete ADV, tuk tuk, tyckel
        labels = labels[labels[:, 0] != 18, :]
        labels = labels[labels[:, 0] != 16, :]
        labels = labels[labels[:, 0] != 14, :]

        # correct the order of the classes
        labels[labels[:, 0] == 15, 0] = 14
        labels[labels[:, 0] == 17, 0] = 15
        labels[labels[:, 0] == 19, 0] = 16

        # one transformend and one not
        if self.acess_numb and not self.val:
            #print(1)
            #augment image
            img = img_augmentor(img)

            # augmaent boxes
            img, labels = rand_flib(img, labels)
            img, labels = rand_rotate(img, labels)

        # upate acess_num
        self.acess_numb = not(self.acess_numb)


        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

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





        if os.path.exists(label_path):


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
            p1_x += pad[1][0]
            p2_x += pad[1][0]
            p3_x += pad[1][0]
            p4_x += pad[1][0]
            p1_y += pad[0][0]
            p2_y += pad[0][0]
            p3_y += pad[0][0]
            p4_y += pad[0][0]

            # Normalize origin for yolo GT to be from 0 to 1
            labels[:, 1] = (p1_x+p2_x+p3_x+p4_x) / (4*padded_w)
            labels[:, 2] = (p1_y+p2_y+p3_y+p4_y) / (4*padded_h)

            # Get height and width after padding, and normalize it
            # normalize using the box diagonal
            diagonal_length = np.sqrt(padded_w**2+padded_h**2)
            val1 = np.sqrt(((p2_x - p1_x) ** 2) + ((p2_y - p1_y) ** 2))
            val2 = np.sqrt(((p3_x - p2_x) ** 2) + ((p3_y - p2_y) ** 2))
            labels[:, 3] = np.min([val1, val2], axis=0) / diagonal_length  # width
            labels[:, 4] = np.max([val1, val2], axis=0) / diagonal_length  # length

            # Normalize theta
            labels[:, 5] = (labels[:, 5] + 180) % 180  #theta[0, 180)
            labels[:, 5] /= 180  #[0,1)
			
			# delete samll objects
            boxes_area = (labels[:, 3]*self.img_shape[0]) * (labels[:, 4]*self.img_shape[0])
            labels = labels[boxes_area > 50]

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 6))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # filled_labels = torch.from_numpy(filled_labels[:,:5])  # to ignore orientation
        filled_labels = torch.from_numpy(filled_labels) # class_id,x,y,w,l,theta

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
