import time
import copy

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage
import torchvision.models as models

from pnasnet.model import NetworkImageNet
from pnasnet.genotypes import PNASNet

from random import randint, seed
from timeit import default_timer

from scipy.misc import imread, imresize
from imgaug import augmenters as iaa

from tensorboardX import SummaryWriter
from utils.train_utils import *


# https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
def check_cuda(model):
    return next(model.parameters()).is_cuda


class Classifier:
    """
        
    """
    def __init__(self, model, num_classes, device='cuda', transform=None):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.transform = transform

        if check_cuda(model) != (device == 'cuda'):
            self.model.to(device)

        model.eval()

    def predict(self, x, probs=False):
        if self.transform is not None:
            x = self.transform(x)

        with torch.no_grad():
            x = x.to(self.device)

            y = self.model(x).detach().cpu()
            if type(y) == tuple:
                y, _ = y
            preds = torch.max(y, 1)[1]

            torch.cuda.empty_cache()
            del x
        if probs:
            return y
        return preds

