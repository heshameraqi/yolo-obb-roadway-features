from __future__ import division, print_function

import numpy as np
import os
import torch

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

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    box = torch.tensor(box)
    clusters = torch.tensor(clusters)
    ious = bbox_iou(box.unsqueeze(0), clusters)
    ious = ious * torch.abs(torch.cos((box[2] - clusters[:, 2])* np.pi / 180))
    return ious.numpy()


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
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
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)
        print(nearest_clusters)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path, img_size, resize_size):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 5)
    """
    result = np.array([[0,0,0]
                       ,[0,0,0]])
    annotations_pathes = [annotation_path +'/'+s for s in os.listdir(annotation_path) if s.endswith('txt')]
    for annotation in annotations_pathes:
        if annotation == "data/classes.txt": continue
        labels =  np.loadtxt(annotation, delimiter=' ', skiprows=1)
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)

        # Get height and width after padding, and normalize it
        # assume producing square image
        paded_size = max(img_size[0], img_size[1])
        val1 = labels[:, 3].copy()
        val2 = labels[:, 4].copy()
        labels[:, 3] = np.min([val1, val2], axis=0) / paded_size  # width
        labels[:, 4] = np.max([val1, val2], axis=0) / paded_size  # length
        labels[:, 3] *= resize_size
        labels[:, 4] *= resize_size

        # delete ADV, tuk tuk, tyckel
        labels = labels[labels[:, 0] != 18, :]
        labels = labels[labels[:, 0] != 16, :]
        labels = labels[labels[:, 0] != 14, :]

        boxes_area = labels[:, 3] * labels[:, 4]
        labels = labels[boxes_area > 50]

        # Normalize theta
        labels[:, 5] = (labels[:, 5] + 180) % 180  # theta[0, 180)

        labels = labels[:, 3:]
        result = np.concatenate([result, labels], axis = 0)

    return result[2:]


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':

    '''
        We use this script to find the anchors and the angels that represent the general case ,
        that represent the oriented BB in the data 

        We define the input image size (which the size in the data used) 
        , and the output image size(which will be used in the training) to normalize them 
        , and convert BBoxes sizes to the images size used in the training .
        '''

    annotation_path = "data/data"
    input_img_size = (2048, 1152)
    ouput_img_size = 416
    anno_result = parse_anno(annotation_path, input_img_size, ouput_img_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{},{}, '.format(anchor[0], anchor[1], anchor[2])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)
