from __future__ import division

import argparse

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.ticker import NullLocator

from PIL import Image
import numpy as np

import random
import os

'''
This script is used to draw the Oriented BBoxes the image and save it.

The ouput images will be save to folder in the same directory called output
'''

parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, default='data', help='path to dataset')
opt = parser.parse_args()
print(opt)

os.makedirs("output", exist_ok=True)

imgs = [opt.image_file + "/" + s for s in os.listdir(opt.image_file) if s.endswith(".jpg")]

fp = open(opt.image_file + "/" + "classes.txt", "r")
classes = fp.read().split("\n")[:-1]
fp.close()

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print("\nSaving images:")
# Iterate through images and save plot of detections
for img_i, img_path in enumerate(imgs):

    print("(%d) Image: '%s'" % (img_i, img_path))

    # Create plot
    img = np.array(Image.open(img_path))
    height, width, depth = img.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(80), height / float(80)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(img)

    labels = np.loadtxt(img_path.replace(".jpg", ".txt"), skiprows=1, delimiter=' ')

    detections = labels[:,0]

    p1_x = labels[:, 1] + labels[:, 3] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 4] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p1_y = labels[:, 2] - labels[:, 3] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 4] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p2_x = labels[:, 1] - labels[:, 3] * np.cos(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 4] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p2_y = labels[:, 2] + labels[:, 3] * np.sin(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 4] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p3_x = labels[:, 1] - labels[:, 3] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 4] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p3_y = labels[:, 2] + labels[:, 3] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 4] * np.sin(np.radians(90 + labels[:, 5])) / 2.0

    p4_x = labels[:, 1] + labels[:, 3] * np.cos(np.radians(labels[:, 5])) / 2.0 - \
           labels[:, 4] * np.cos(np.radians(90 + labels[:, 5])) / 2.0
    p4_y = labels[:, 2] - labels[:, 3] * np.sin(np.radians(labels[:, 5])) / 2.0 + \
           labels[:, 4] * np.sin(np.radians(90 + labels[:, 5])) / 2.0


    # Draw bounding boxes and labels of detections
    if detections is not None:

        unique_labels = np.unique(detections)
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for i, (cls_pred, x, y, w, le, theta) in enumerate(labels):

            #if cls_pred != 0: continue  #to display specific objects

            print("\t+ Label: %s, angel : %.5f" % (classes[int(cls_pred)], theta))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # Create a polygon patch
            verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
            path = Path(verts, codes)
            obbox = patches.PathPatch(path, linewidth=3, edgecolor=color, facecolor='none')

            # Add the bbox to the plot
            ax.add_patch(obbox)
            # Add label
            plt.text(
                x,
                y,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig("output/%s.png" % (img_path.replace(opt.image_file + "/", "")), bbox_inches="tight", pad_inches=0.0)
    #plt.show()
    plt.close()



