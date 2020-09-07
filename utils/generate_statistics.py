from __future__ import division, print_function

import numpy as np
import os
import torch
from collections import defaultdict
from utils.utils import load_classes
import matplotlib.pyplot as plt


#  For anchor statistics

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = - box1[:, 0] / 2, box1[:, 0] / 2
    b1_y1, b1_y2 = - box1[:, 1] / 2, box1[:, 1] / 2
    b2_x1, b2_x2 = - box2[:, 0] / 2, box2[:, 0] / 2
    b2_y1, b2_y2 = - box2[:, 1] / 2, box2[:, 1] / 2

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


def iou(box, clusters, with_angel = False):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    if(with_angel):
        box = torch.tensor(box)
        clusters = torch.tensor(clusters)
        ious = bbox_iou(box.unsqueeze(0), clusters)
        ious = ious * torch.abs(torch.cos(box[2] - clusters[:, 2]))
        return ious.numpy()

    else:
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)
        return iou_


def kmeans(boxes, k, dist=np.median, with_angel = False):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters, with_angel)

        nearest_clusters = np.argmin(distances, axis=1)
        print(nearest_clusters)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def avg_iou(boxes, clusters, with_angel = False):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters, with_angel)) for i in range(boxes.shape[0])])


def parse_anno(annotation_path, with_angel = False):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 5)
    """
    result = np.array([[0,0,0]
                       ,[0,0,0]])
    annotations_pathes = [annotation_path +'/'+s for s in os.listdir(annotation_path) if s.endswith('txt')]
    for annotation in annotations_pathes:
        if annotation == annotation_path + '/' + "classes.txt": continue
        labels =  np.loadtxt(annotation, delimiter=' ', skiprows=1)
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)
        labels = labels[..., 3:]
        labels[:,2] += 180
        labels[:, 2] %= 180
        #print(labels.shape)
        result = np.concatenate([result, labels], axis = 0)
    if with_angel:
        return result[2:]
    else :
        return result[2:, :-1] # Delete angel part


def get_kmeans(anno, cluster_num=9, with_angel = False):

    anchors = kmeans(anno, cluster_num, with_angel=with_angel)
    ave_iou = avg_iou(anno, anchors, with_angel=with_angel)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


def anchors_Without_angels(annotation_path = "train"):
    '''
    get anchors statistics without angel
    :param annotation_path: the path of traning labels
    :return: print the anchors data
    '''
    annotation_path = annotation_path
    anno_result = parse_anno(annotation_path, with_angel=False)
    anchors, ave_iou = get_kmeans(anno_result, 9, with_angel=False)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)


def anchors_with_angel(annotation_path = "train"):
    '''
    get anchors statistics with angel
    :param annotation_path: the path of traning labels
    :return: print the anchors data
    '''
    annotation_path = annotation_path
    anno_result = parse_anno(annotation_path, with_angel=True)
    anchors, ave_iou = get_kmeans(anno_result, 9, with_angel=True)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{},{}, '.format(anchor[1], anchor[0], anchor[2])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)




#  For data statistics


def parse_anno(annotation_path):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 6)
    """
    result = np.array([[0,0,0,0,0,0]
                       ,[0,0,0,0,0,0]])
    annotations_pathes = [annotation_path +'/'+s for s in os.listdir(annotation_path) if s.endswith('txt')]
    for annotation in annotations_pathes:
        if annotation == annotation_path +'/'+ "classes.txt" : continue
        labels =  np.loadtxt(annotation, delimiter=' ', skiprows=1)
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)
        w, h = labels[..., 0].copy(), labels[..., 1].copy()
        labels[..., 0] = np.min([w, h], axis=0)
        labels[..., 1] = np.max([w, h], axis=0)
        #print(labels.shape)
        result = np.concatenate([result, labels], axis = 0)

    return result[2:]


def draw_hist(x, value_name,  bins):
    plt.hist(x, bins = bins)
    plt.ylabel('No of times')
    plt.xlabel(value_name)
    plt.show()

def generate_angel_statistics(annotation_path = "train"):
    print("angel Statistics\n")
    labels = parse_anno(annotation_path)
    draw_hist(labels[:, 5], "theta", 18)
    theta_appers = defaultdict(int)
    for theta in labels[:,5]:
        theta_appers[int(theta)] += 1
    sort =  sorted(theta_appers.items(), key = lambda x:(x[1],x[0]), reverse= True)
    for ang, val in sort[:]:
        print(f"{ang} appeard {val}")


def generate_classes_statistics(annotation_path = "train"):
    print("\n\n classes Statistics\n")

    labels = parse_anno(annotation_path)
    # for the classes
    classes = np.array(load_classes("D:/ML/Valeo/train/classes.txt"))
    classes_appears = defaultdict(int)
    for cls in labels[:,0]:
        classes_appears[cls] += 1
    x =[]
    for xx in classes_appears.keys() : x.append(int(xx))
    plt.bar(classes[x], classes_appears.values())
    plt.show()
    sort = sorted(classes_appears.items(), key = lambda x:(x[1],x[0]), reverse=True)
    for cls, val in sort:
        print(f"{classes[int(cls)]} appears {val}")

