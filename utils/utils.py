from __future__ import division
import math
import time
import torch
import torch.nn as nn
import tqdm

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import shapely.geometry
import shapely.affinity

import matplotlib.pyplot as plt
from descartes import PolygonPatch
import matplotlib.patches as patches


class OBB:  # Takes angle in degrees
    def __init__(self, cx, cy, w, le, angle):
        self.cx = cx
        self.cy = -cy  # minus because y is defined downside
        self.w = le
        self.le = w
        self.angle = angle

    def get_contour(self):
        c = shapely.geometry.box(-self.w/2.0, -self.le/2.0, self.w/2.0, self.le/2.0)
        rc = shapely.affinity.rotate(c, self.angle.copy())
        return shapely.affinity.translate(rc, self.cx.copy(), self.cy.copy())

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def union(self, other):
        return self.get_contour().union(other.get_contour())

    def iou(self, other, visualize=False):
        intersect_area = self.intersection(other).area
        union_area = self.union(other).area

        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.add_patch(PolygonPatch(self.get_contour(), fc='#990000', alpha=0.7))
            ax.add_patch(PolygonPatch(other.get_contour(), fc='#000099', alpha=0.7))
            ax.add_patch(PolygonPatch(self.intersection(other), fc='#009900', alpha=1))
            # plt.show()
            plt.show(block=False)

        return intersect_area / (union_area + 1e-16)
		
		
def visualize_data(imgs, targets):
    for sample_id in range(imgs.shape[0]):
        image = np.transpose(imgs[sample_id].numpy(), (1, 2, 0))
        labels = targets[sample_id].numpy()

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.collections as collections
        from matplotlib.path import Path
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(image)

        # denormalize x,y
        labels[:, 1] *= image.shape[0]
        labels[:, 2] *= image.shape[1]

        # denormalize w,l
        diagonal_length = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
        labels[:, 3] *= diagonal_length
        labels[:, 4] *= diagonal_length

        # denormalize theta
        labels[:, 5] *= 90.

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
            patches.append(mpl.patches.PathPatch(path, linewidth=1, edgecolor='r', facecolor='none'))
            ax.text(verts[0][0], verts[0][1], classes[int(labels[i][0])], fontsize=6,
                    bbox=dict(edgecolor='none', facecolor='white', alpha=0.8, pad=0.))
        ax.add_collection(collections.PatchCollection(patches, match_original=True))
        # plt.show(block=False)
        plt.show()


def to_cpu(tensor):
    return tensor.detach().cpu()

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def get_batch_statistics(outputs, targets, iou_threshold):
    """
    Compute true positives, predicted scores and predicted labels per sample
    output (N)(x,y, w, le, theta, score, pred)
    targets (N)(label, x,y, w, le, theta)
    """

    batch_metrics = []
    for sample_i in range(len(outputs)):
        annotations = to_cpu(targets[sample_i][targets[sample_i][:, -2] > 0]).numpy()#TODO # clean the zeros palceholders from dataset.py
        target_labels = annotations[:, 0] if len(annotations) else []

        if outputs[sample_i] is None:
            continue

        output = to_cpu(outputs[sample_i]).numpy()
        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]
        # this is done by finding the IOU of each prediction with all the tagets
        # and the biggest IOU is assigned to the prediction
        # if the IOU is bigger than the threshould then it's considered to be TP sampel
        true_positives = np.zeros(pred_boxes.shape[0])
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            # unnormalize target output
            target_boxes[:, :2] *= 416
            target_boxes[:, 2:4] *= 416 * np.sqrt(2)
            target_boxes[:, 4] *=90

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                ious = bbox_iou_obb_H(np.expand_dims(pred_box, 0), target_boxes).unsqueeze(0).numpy()
                iou, box_index = ious.max(1), ious.argmax(1)
                if iou >= iou_threshold and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bbox_iou_obb(box1, box2, visualize=False):  # box format is: x,y,w,l,theta(degrees)
    """
    Returns the ArIoU of two bounding boxes
    """
    ious = bbox_iou(box1, box2, x1y1x2y2= False)
    ious = ious * torch.abs(torch.cos((box1[:, 4] - box2[:, 4]) * np.pi / 180))
    return ious

def bbox_iou_obb_H(box1, box2, visualize=False):  # box format is: x,y,w,l,theta(degrees)
    """
    Returns the IoU of two bounding boxes
    """
    ious = torch.empty(box2.shape[0])
    r1 = OBB(box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3], box1[:, 4])
    for i in range(box2.shape[0]):
        r2 = OBB(box2[i, 0], box2[i, 1], box2[i, 2], box2[i, 3], box2[i, 4])
        ious[i] = r1.iou(r2, visualize=visualize)
    return ious

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.

    prediction shape is: 16 (batch size) X 10647 (52X52+26X26+13X13 feature maps outputs from YOLOv3 X 3 anchors)
                                         X 26 (x,y,w,l,theta,objectiveness, 20 classes)

    Returns detections with shape:
        (x1, y1, w, le, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, lenght) to (x1, y1, x2, y2)
    #box_corner = prediction.new(prediction.shape)
    #box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    #box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    #box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    #box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    #prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 5] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 6 : 6 + num_classes], 1, keepdim=True) # TODO:
        # Detections ordered as (x, y, w, l, theta, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :6], class_conf.float(), class_pred.float()), 1) #TODO:
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 5], descending=True) #TODO:
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou_obb_H(max_detections[-1].cpu().numpy(), detections_class[1:].cpu().numpy(), visualize=False)#TODO:
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):

    # target size is batch_size X max_number_of_objects_in_image X 6 (class_id,x,y,w,l,theta) (normalized GT)
    # pred_boxes size is nBatch X nAnchors X featuremap X 5 (x,y,w,l,theta), after demoralizing with respect to anchors
    # pred_cls size is nBatch X nAnchors X featuremap X nClasses
    # pred_conf size is nBatch X nAnchors X featuremap X 1
    # anchors size is nAnchors X 3 (w,l,theta)

    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    
    # each cell corresponds to predicted 3 boxes. each is represented by offset from the 3 anchor boxes
    # each cell has a ground-truth and 3 anchors, mask is all zeros except for the anchor nearest to ground-truth
    mask = torch.zeros(nB, nA, nG, nG) # the best anchor box
    conf_mask = torch.ones(nB, nA, nG, nG) # the Anchor boxes that will be trained on objectivness
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    tl = torch.zeros(nB, nA, nG, nG)
    ttheta = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0 # will store the number of correctly detected objects in the batch out of nGT
    for b in range(nB): # For all sample in the batch
        for t in range(target.shape[1]): # For all the GT objects in this sample
            if target[b, t].sum() == 0: # all zeros, means no object (TODO:should be break instead?)
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * np.sqrt(2)*nG # we also mutiply by np.sqrt(2) as we scaled by diagonal 
            gl = target[b, t, 4] * np.sqrt(2)*nG
            gtheta = target[b, t, 5] * 90

            # Select anchor box with the most similar shape to this object
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gl, gtheta])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            # TODO Match the Anchor boxes without angel
            anch_ious = bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False)
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)

            # Get nearest anchor to GT, then corresponding prediction
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gl, gtheta])).unsqueeze(0)
            # Get the best prediction
            # TODO : why pred_boxes[b, best_n, gj, gi] 3 boxes seems sorted in terms of area in start of training?
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            
            # Correct or not, and return GT of classes, objectiveness, x,y,w,l,theta
            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            target_label = int(target[b, t, 0])
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

            # One-hot encoding of label
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height (if equal to best anchor give 0, positive if bigger than best anchor, negative
            # otherwise)
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            tl[b, best_n, gj, gi] = math.log(gl / anchors[best_n][1] + 1e-16)
            
            # theta
            ttheta[b, best_n, gj, gi] = gtheta / 90
            
            # Mask for yolo loss
            mask[b, best_n, gj, gi] = 1
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            conf_mask[b, best_n, gj, gi] = 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, tl, ttheta, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])
