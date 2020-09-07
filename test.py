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
parser.add_argument("--image_folder", type=str, default="train", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
#parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_90.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="train/classes.txt", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.1, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
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
dataset = ListDataset(opt.image_folder, val=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad():
        outputs = model(imgs)
        outputs = non_max_suppression(outputs, num_classes, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
        torch.set_printoptions(profile="full")
        print(outputs)
        torch.set_printoptions(profile="default")

    for output, annotations in zip(outputs, targets):

        all_detections.append([np.array([]) for _ in range(num_classes)])
        if output is not None:
            # Get predicted boxes, confidence scores and labels
            pred_boxes = output[:, :6].cpu().numpy() #TODO:
            scores = output[:, 5].cpu().numpy() #TODO:
            pred_labels = output[:, -1].cpu().numpy()

            # Order by confidence
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        all_annotations.append([np.array([]) for _ in range(num_classes)])
        if any(annotations[:, -2] > 0): #TODO

            annotation_labels = annotations[annotations[:, -2] > 0, 0].numpy()
            _annotation_boxes = annotations[annotations[:, -2] > 0, 1:]

            # TODO Rescale the target image
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] * 416
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] * 416
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] * np.sqrt(2) * 416
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] * np.sqrt(2) * 416
            annotation_boxes[:, 4] = _annotation_boxes[:, 4] * 90


            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]


    break

# this is done by finding the IOU of each prediction with all the tagets
# and the biggest IOU is assigned to the prediction
# if the IOU is bigger than the threshould then it's considered to be TP sampel

average_precisions = {}
for label in range(num_classes):
    true_positives = []
    scores = []
    num_annotations = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.append(0)
                continue

            overlaps = bbox_iou_obb(np.expand_dims(bbox, axis=0), annotations).unsqueeze(0).numpy()
            #print(overlaps)

            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # no annotations -> AP for this class is 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    true_positives = np.array(true_positives)
    false_positives = np.ones_like(true_positives) - true_positives
    # sort by score
    indices = np.argsort(-np.array(scores))
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision

print("Average Precisions:")
for c, ap in average_precisions.items():
    print(f"+ Class '{c}' - AP: {ap}")

mAP = np.mean(list(average_precisions.values()))
print(f"mAP: {mAP}")
