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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=201, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/data", help="path to dataset")
parser.add_argument("--label_files", type=str, default="train.txt", help="files of the names of the annotations")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
#parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_weights.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/data/classes.txt", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda
print(f"Cuda is working? {cuda}")

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# get taining path
train_path = opt.image_folder

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)

# Load the weights of the model Darknet weights
model.apply(weights_init_normal)
model_dict = model.state_dict() # state of the current model 
pretrained_dict = torch.load(opt.weights_path) # state of the pretrained model
pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('81' not in k) and ('93' not in k) and ('105' not in k)} # remove the classifier from the state
classifier_dict = {k: v for k, v in model_dict.items() if ('81' in k) or ('93' in k) or ('105' in k)} # get the classifier weight from new model
pretrained_dict.update(classifier_dict)
model_dict.update(pretrained_dict)  # update without classifier
model.load_state_dict(pretrained_dict)  # the model know has the wights of the model without angel but the classifier part is intialized


if cuda:
    model = model.cuda()

model.train()

# Get dataloader (train_path is a path of file with list of all train and validation images files)
# theta required in degrees
dataloader = torch.utils.data.DataLoader(
    ListDataset(opt.label_files, train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# filter the parameters that require grad
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))


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


for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, le %f, theta %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["le"],
                model.losses["theta"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
