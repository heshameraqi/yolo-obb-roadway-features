import numpy as np
import os
from utils.utils import load_classes
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_anno(annotation_path, num_classes):
    """
    Read annotations from text file and convert it ti a numpy array
    :param annotation_path: Text
    :return: result: (N, 6)
    """

    annotations_files = [s for s in os.listdir(annotation_path) if s.endswith('txt')]
    annotations = defaultdict(str)
    objects_appear = np.zeros(20)
    for annotation in annotations_files:
        if annotation ==  "classes.txt" : continue
        labels =  np.loadtxt(annotation_path +'/'+ annotation, delimiter=' ', skiprows=1)
        appear = np.zeros(num_classes) # classes number
        if(len(labels.shape) == 1): labels = np.expand_dims(labels, axis = 0)
        appear[[int(x) for x in labels[:,0]]] = 1
        objects_appear += appear
        annotations[annotation] = appear
        #print(labels.shape)
        objects_order = np.argsort(objects_appear)
    return annotations, objects_appear, objects_order

if __name__ == '__main__':

    '''
    This script used to split the data set (20% val, 80% training), randomly and also balancing the class appearance.
    Every class should appear with the same percentage as the split. 
    '''

    classes = np.array(load_classes("test/classes.txt"))
    imgs, objects_appear, objects_order =  parse_anno("test", 20)
    classes_appears = defaultdict(int)
    train = []
    val = []
    imgs_keys = list(imgs.keys())
    for cls in objects_order:
        val_pres = int(objects_appear[cls] * .2)
        count = 0
        for img in imgs_keys:
            if imgs[img][cls] == 1:
                if count <= val_pres: val.append(img)
                else : train.append(img)
                objects_appear -= imgs[img]
                imgs_keys.remove(img)
                count+=1
    file = open("validation2.txt", "w")
    for v in val[:343]:
        file.write(v+"\n")

    val = val[:343]
    imgs_names = list(imgs.keys())
    file2 = open("train2.txt", "w")
    for name in imgs_names:
        if (name not in val): file2.write(name + "\n")







