from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from matplotlib.path import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='train', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='checkpoints/yolov3_ckpt_90.pth', help='path to weights file')
parser.add_argument('--class_path', type=str, default='train/classes.txt', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda
print(cuda)

os.makedirs('output', exist_ok=True)

# Get classes number
classes = load_classes(opt.class_path)
num_classes = len(classes)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_state_dict(torch.load(opt.weights_path))

if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode

dataloader = DataLoader(ListDataset(opt.image_folder, img_size=opt.img_size, val = True ),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs, _) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, num_classes, opt.conf_thres, opt.nms_thres)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)
    break


# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print("\nSaving images:")
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    labels = detections[:,-1]
    detections = detections.cpu().numpy()

    p1_x = detections[:, 0] + detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 + \
           detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
    p1_y = detections[:, 1] - detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 - \
           detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

    p2_x = detections[:, 0] - detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 + \
           detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
    p2_y = detections[:, 1] + detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 - \
           detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

    p3_x = detections[:, 0] - detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 - \
           detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
    p3_y = detections[:, 1] + detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 + \
           detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

    p4_x = detections[:, 0] + detections[:, 3] * np.cos(np.radians(detections[:, 4])) / 2.0 - \
           detections[:, 2] * np.cos(np.radians(90 + detections[:, 4])) / 2.0
    p4_y = detections[:, 1] - detections[:, 3] * np.sin(np.radians(detections[:, 4])) / 2.0 + \
           detections[:, 2] * np.sin(np.radians(90 + detections[:, 4])) / 2.0

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    # Rescale coordinates to original dimensions
    p1_x = ((p1_x - pad_x // 2) / unpad_w) * img.shape[1]
    p2_x = ((p2_x - pad_x // 2) / unpad_w) * img.shape[1]
    p3_x = ((p3_x - pad_x // 2) / unpad_w) * img.shape[1]
    p4_x = ((p4_x - pad_x // 2) / unpad_w) * img.shape[1]
    p1_y = ((p1_y - pad_y // 2) / unpad_h) * img.shape[0]
    p2_y = ((p2_y - pad_y // 2) / unpad_h) * img.shape[0]
    p3_y = ((p3_y - pad_y // 2) / unpad_h) * img.shape[0]
    p4_y = ((p4_y - pad_y // 2) / unpad_h) * img.shape[0]

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = labels.cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for i, (x, y, w, le, theta, conf, cls_conf, cls_pred) in enumerate(detections):
            print("\t+ Label: %s, Conf: %.5f, angel : %.5f" % (classes[int(cls_pred)], cls_conf.item(), theta))

            y = ((y - pad_y // 2) / unpad_h) * img.shape[0]
            x = ((x - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # Create a polygon patch
            verts = [(p1_x[i], p1_y[i]), (p2_x[i], p2_y[i]), (p3_x[i], p3_y[i]), (p4_x[i], p4_y[i]), (0., 0.), ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
            path = Path(verts, codes)
            obbox = patches.PathPatch(path, linewidth=1, edgecolor=color, facecolor='none')

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
    plt.savefig("output/%d.png" % (img_i), bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

    '''
    # for labels to text file
        detections[:, 0] = ((p1_x + p2_x + p3_x + p4_x) / 4)
        detections[:, 1] = ((p1_y + p2_y + p3_y + p4_y) / 4)

        dim1 = np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
        dim2 = np.sqrt((p3_x - p2_x) ** 2 + (p3_y - p2_y) ** 2)
        detections[:, 3] = np.min([dim1, dim2], axis=0)  # width
        detections[:, 2] = np.max([dim1, dim2], axis=0)  # lenght

        out = np.concatenate([detections[:, -1:], detections[:, :5]], axis=1)

        np.savetxt("meta_before/" + path.replace(".png", ".txt").replace(".jpg", ".txt").replace(
            '/content/drive/My Drive/ Cairo-Alex Disert Road/', ''), out, fmt='%.2f', delimiter=' ', header='YOLO_OBB')
    '''