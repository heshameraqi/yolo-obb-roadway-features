import glob
import random
import os
import numpy as np

import torch

import random
from matplotlib.ticker import NullLocator
from utils.utils import *

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

from imgaug import augmenters as iaa

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
            img_new = np.array(transforms.functional.rotate(Image.fromarray(np.uint8(img)), angel, expand=True), dtype=np.uint8)

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



def visualize_data_image(imgs, targets):
    image = imgs
    labels = targets

    classes = load_classes("/media/user/76E6B044E6B00701/OmarElezaby/data/classes.txt")

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.collections as collections
    from matplotlib.path import Path
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(image)

    p1_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p1_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p2_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p2_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p3_x = labels[:, 1] - labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p3_y = labels[:, 2] + labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p4_x = labels[:, 1] + labels[:, 4] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 3] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p4_y = labels[:, 2] - labels[:, 4] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 3] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    patches = []
    for i in range(labels.shape[0]):
        if not np.any(labels[i]):  # objects in image finished before max_objects
            break
        verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
        codes = [Path.MOVETO,        Path.LINETO,        Path.LINETO,        Path.LINETO,        Path.CLOSEPOLY, ]
        path = Path(verts, codes)
        patches.append(mpl.patches.PathPatch(path, linewidth=3, edgecolor='r', facecolor='none'))
        ax.text(labels[i, 1], labels[i, 2], classes[int(labels[i][0])], fontsize=6,
                            bbox=dict(edgecolor='none', facecolor='white', alpha=0.8, pad=0.))
    ax.add_collection(collections.PatchCollection(patches, match_original=True))
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig("output/Orginal.png", bbox_inches="tight", pad_inches=0.0)

    plt.show()




if __name__ == '__main__':
    '''
    This script is used to see the effect of every part of data augmentation ,
    and ensure that data augmentation functions working properly 
    '''

    # augmentor objects
    img_augmentor = Image_augmentation()
    rand_flib = HorizinatFlib()
    rand_rotate = Random_rotate()

    # load image
    img_path = "data/867363024672796_30066913_31136786_20180825063223_0_M_a9756477-e1a1-423b-9969-46eeffe1838b.jpg"
    img = np.array(Image.open(img_path))

    # load labels
    label_path = "data/867363024672796_30066913_31136786_20180825063223_0_M_a9756477-e1a1-423b-9969-46eeffe1838b.txt"
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

    # augment image
    #img = img_augmentor(img)

    # augmaent boxes
    #img, labels = rand_flib(img, labels)
    #img, labels = rand_rotate(img, labels)

    w, h = labels[:, 3].copy(), labels[:, 4].copy()
    labels[:, 3], labels[:, 4] = h, w

    visualize_data_image(img, labels)