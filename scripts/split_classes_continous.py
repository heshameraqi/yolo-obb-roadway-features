import numpy as np
from collections import defaultdict
import os
from utils.utils import load_classes
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from PIL import Image

#find best split
best_error = 1000
best_start = ""
best_start_indx = -1

data_all = open("data.txt","r")
data_all_names = [s.replace("\n","") for s in data_all.readlines()]
data_folder = "data/data"
val_size = int(len(data_all_names) * 0.2)

result = np.array([[0, 0, 0, 0, 0, 0]
                      , [0, 0, 0, 0, 0, 0]])
for annotation in data_all_names:
    if annotation == "classes.txt": continue
    labels = np.loadtxt(data_folder + '/' + annotation, delimiter=' ',
                        skiprows=1)
    if (len(labels.shape) == 1): labels = np.expand_dims(labels, axis=0)
    # delete ADV, tuk tuk, tyckel
    labels = labels[labels[:, 0] != 18, :]
    labels = labels[labels[:, 0] != 16, :]
    labels = labels[labels[:, 0] != 14, :]

    # correct the order of the classes
    labels[labels[:, 0] == 15, 0] = 14
    labels[labels[:, 0] == 17, 0] = 15
    labels[labels[:, 0] == 19, 0] = 16

    result = np.concatenate([result, labels], axis=0)

labels_data = result[2:].copy()


for i, img in enumerate(data_all_names):
    result = np.array([[0, 0, 0, 0, 0, 0]
                          , [0, 0, 0, 0, 0, 0]])
    if(len(data_all_names) - i) < val_size : break
    data_chunck = data_all_names[i:i+val_size]
    for annotation in data_chunck:
        if annotation == "classes.txt": continue
        labels = np.loadtxt(data_folder + '/' + annotation, delimiter=' ', skiprows=1)
        if (len(labels.shape) == 1): labels = np.expand_dims(labels, axis=0)
        # delete ADV, tuk tuk, tyckel
        labels = labels[labels[:, 0] != 18, :]
        labels = labels[labels[:, 0] != 16, :]
        labels = labels[labels[:, 0] != 14, :]

        # correct the order of the classes
        labels[labels[:, 0] == 15, 0] = 14
        labels[labels[:, 0] == 17, 0] = 15
        labels[labels[:, 0] == 19, 0] = 16

        result = np.concatenate([result, labels], axis=0)
    labels_train = result[2:]


    # for the classes
    #classes = np.array(load_classes("data/data/classes.txt"))
    classes_appears_data = defaultdict(int)
    classes_appears_train = defaultdict(int)
    for cls in labels_train[:, 0]:
        classes_appears_train[cls] += 1
    for cls in labels_data[:, 0]:
        classes_appears_data[cls] += 1
    x = []
    for xx in classes_appears_train.keys(): x.append(int(xx))
    # plt.bar(classes[x], classes_appears_train.values())
    # plt.show()
    sort = sorted(classes_appears_train.items(), key=lambda x: (x[1], x[0]), reverse=True)
    error_count = 0
    for cls, val in sort:
        error_count += abs(.2 - val / classes_appears_data[cls]) if ((abs(.2 - val) / classes_appears_data[cls]) > .03) else 0
    if error_count < best_error:

        print(error_count)
        best_error = error_count
        best_start = img
        best_start_indx = i


print(f"{best_start} have error {best_error}")
file = open("validation.txt", "w")

for v in data_all_names[best_start_indx:best_start_indx+val_size]:
    file.write(v+"\n")

file2 = open("train.txt", "w")
for name in data_all_names:
    if (name not in val): file2.write(name + "\n")