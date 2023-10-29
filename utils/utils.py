import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split
from torch.utils import data 
from torch.autograd import Variable 
import torch
import torch.nn as nn
from torch.nn import init

import torch
from torchvision import datasets,transforms
import torch.utils.data as Dataset

import joblib
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable 
import scipy.stats

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.utils.model_targets import SoftmaxOutputTarget
import cv2
import os.path as op
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. Copy from pytorchtools"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        print(self.path)
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss
        

def grad_cam_plot(model, layerID, image):
    class MaxOutputTarget(SoftmaxOutputTarget):
        def __call__(self, output):
            return torch.max(output)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    model.eval()

    if layerID == 1:
        target_layers = [model.layer1]
    elif layerID == 2:
        target_layers = [model.layer1, model.layer2]
    elif layerID == 3:
        target_layers = [model.layer1, model.layer2, model.layer3]

    # layers = [model.conv1, model.conv2, model.conv3]
    # target_layers = layers[layerID-1]

    #     images = np.array((pickle.load(open('C:/Users/M/Desktop/array_list/array_list.pkl','rb'))), dtype=np.float32).reshape((-1, 64, 60))

    im_tensor = torch.from_numpy(image.astype("float32"))
    im_tensor = im_tensor.reshape(1, 1, 64, 60)

    input_tensor = im_tensor
    input_tensor = input_tensor.reshape(1,1,64,60)

    rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb_img = rgb_img.astype("float32")

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(torch.cuda.is_available()))

    targets = [MaxOutputTarget()]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # [batch, 224,224]

    grayscale_cam = grayscale_cam[0].astype("float32")
    visualization = show_cam_on_image(rgb_img/255, grayscale_cam)  # (224, 224, 3)
    images = np.hstack((np.uint8(255 * rgb_img), visualization))

    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    # images_show = np.hstack((np.uint8(255 * rgb_img), visualization_bgr))
#     images_show = np.hstack((np.uint8(rgb_img), visualization_bgr))
    
#     plt.imshow(visualization_bgr)
    # cv2.imwrite(f'CNN_classification_{idx}layer{layerID}.jpg', images)
    return visualization_bgr


def initialize_parameters(model, nonlinearity):
    for module in model.modules():
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity=nonlinearity)
    return model