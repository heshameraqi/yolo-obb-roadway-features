from __future__ import division

from utils.utils import *
from utils.datasets import *

from torch.utils.data import DataLoader


def visualize_data_batch(imgs, targets):
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
        labels[:, 5] *= 180.

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

if __name__ == '__main__':
    '''
    This script is used to plot the output form data-set to see output images and labels from data-set class
    and check if the data-set class is working properly  
    '''

    labels_files = "detect_imgs.txt"
    data_folder = "data/data"
    data = ListDataset(labels_files, data_folder)
    classes = load_classes(data_folder+"/classes.txt")
    print(len(classes))

    data_loder = DataLoader(data, batch_size=8,)
    for _, img, labels in data_loder:

        visualize_data_batch(img, labels)
