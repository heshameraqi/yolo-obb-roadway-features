from __future__ import division

from models import *
from utils.utils import *
from utils.Logger import *
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
parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/data", help="path to dataset")
parser.add_argument("--label_files", type=str, default="train.txt", help="files of the names of the annotations")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
#parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_weights.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/data/classes.txt", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)


# Function used to evalute mAP during training
def evaluate(model, path, label_Files, iou_thres, conf_thres, nms_thres, img_size, batch_size, sampels_num, num_classes):
    model.eval()

    # Get dataloader
    dataset = ListDataset(label_Files, path, img_size=img_size, val=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += [label[0] for sample in targets for label in sample if label[-2] > 0 ]

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres, num_classes=num_classes)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        if batch_i * batch_size >= sampels_num: break

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class



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

# load from weight file
if "checkpoints" not in opt.weights_path :
    #print(1)
    # load the weights of the model Darknet weights
    model.apply(weights_init_normal)
    model_dict = model.state_dict() # state of the current model
    pretrained_dict = torch.load(opt.weights_path) # state of the pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('81' not in k) and ('93' not in k) and ('105' not in k)} # remove the classifier from the state
    classifier_dict = {k: v for k, v in model_dict.items() if ('81' in k) or ('93' in k) or ('105' in k)} # get the classifier weight from new model
    pretrained_dict.update(classifier_dict)
    model_dict.update(pretrained_dict)  # update without classifier
    model.load_state_dict(pretrained_dict)  # the model know has the wights of the model without angel but the classifier part is intialized
# load from checkpoint
else :
    model.load_state_dict(torch.load(opt.weights_path))

if cuda:
    model = model.cuda()

model.train()

# Get dataloader (train_path is a path of file with list of all train and validation images files)
# theta required in degrees
dataloader = torch.utils.data.DataLoader(
    ListDataset(opt.label_files, train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# filter the parameters that require grad
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

logger = Logger("logs")

for epoch in range(opt.epochs):
    model.train()
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # for logs steps
        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, le %f, sin %f, cos %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["le"],
                model.losses["sin"],
                model.losses["cos"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        # save Losses to the logger file
        tensorboard_log = []
        for loss_name, value in model.losses.items():
            tensorboard_log += [(loss_name, value)]
        tensorboard_log += [("Total Loss", loss.item())]
        logger.list_of_scalars_summary(tensorboard_log, batches_done)

    model.seen += imgs.size(0)

    if epoch % opt.evaluation_interval == 0:
        print(f"\n---- Epoch_num {epoch}----\n")
        
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=train_path,
            label_Files="train.txt",
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=4,
            sampels_num=400,
            num_classes=len(classes),
        )

        # add to logger file
        evaluation_metrics = [
            ("train_precision", precision.mean()),
            ("train_recall", recall.mean()),
            ("train_mAP", AP.mean()),
            ("train_f1", f1.mean()),
        ]
        for i, c in enumerate(ap_class):
            evaluation_metrics += [(f"+ Class '{c}' ({classes[c]}_training", AP[i])]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

        print("Average Precisions on training:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]}")

        print(f"Training mAP: {AP.mean()}")

        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=train_path,
            label_Files="val.txt",
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=4,
            sampels_num=200,
            num_classes=len(classes),
        )

        # add to logger file
        evaluation_metrics = [
            ("Val_precision", precision.mean()),
            ("Val_recall", recall.mean()),
            ("Val_mAP", AP.mean()),
            ("Val_f1", f1.mean()),
        ]
        for i, c in enumerate(ap_class):
            evaluation_metrics += [(f"+ Class '{c}' ({classes[c]}_Val", AP[i])]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

        print("Average Precisions on Val:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
