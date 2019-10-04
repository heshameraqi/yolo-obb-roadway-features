from __future__ import division

from models import *
import torch

'''
should be run before changing model arch to mantain the weights positions
This script is to change the weghts format to torch state directory to manipulate it
'''


# pathes
model_config_path = "config/yolov3.cfg"
weights_path = "weights/yolov3.weights"

# Initiate model
model = Darknet(model_config_path)
model.load_weights(weights_path)
torch.save(model.state_dict(), f"yolov3_weights.pth")