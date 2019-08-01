from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/data", help="path to dataset")
parser.add_argument("--label_files", type=str, default="val.txt", help="files of the names of the annotations")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
#parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_195.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/data/classes.txt", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda
print(cuda)

# Get classes number
classes = load_classes(opt.class_path)
num_classes = len(classes)

# Initiate model
model = Darknet(opt.model_config_path)
model.load_state_dict(torch.load(opt.weights_path))

if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
dataset = ListDataset(opt.label_files, opt.image_folder, val=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, pin_memory=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

labels = []
sample_metrics = []  # List of tuples (TP, confs, pred)
for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    # Extract labels
    labels += [label[0] for sample in targets for label in sample]

    imgs = Variable(imgs.type(Tensor), requires_grad=False)

    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres, num_classes= num_classes)

    sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=opt.iou_thres)

# Concatenate sample statistics
true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

print("Average Precisions:")
for i, c in enumerate(ap_class):
    print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]}")

print(f"mAP: {AP.mean()}")
