from __future__ import division, print_function

import numpy as np
import os

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


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


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


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
    result = np.array([[0,0]
                       ,[0,0]])
    annotations_pathes = [annotation_path +'/'+s for s in os.listdir(annotation_path) if s.endswith('txt')]
    for annotation in annotations_pathes:
        if annotation == "data/data/classes.txt": continue
        labels =  np.loadtxt(annotation, delimiter=' ', skiprows=1)
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)

        # resize width and length (assuming a square imageS)

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
        labels = labels[boxes_area > 100]

        labels = labels[:, 3:-1]
        #print(labels)
        #print(labels.shape)
        result = np.concatenate([result, labels], axis=0)

    return result[2:]


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':
    '''
    This script used to find the anchor sizes that will used in the training based on the data.
    Using k-means we find the anchor boxes that represent the general case of the sizes od bound boxes.
    
    We define the input image size (which the size in the data used)
    and the output image size(which will be used in the training) to normalize them 
    and convert BBoxes sizes to the images size used in the training .
    '''
    annotation_path = "data/data"
    input_img_size = (2048, 1152)
    ouput_img_size = 416
    anno_result = parse_anno(annotation_path, input_img_size, ouput_img_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)
