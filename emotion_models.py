import torchvision.models as models
import torch.nn as nn


def resnet_18():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    #return model
    return nn.Sequential(model, nn.ReLU(), nn.Softmax())


def pnasnet(num_conv_filters=128, num_cells=10):
    from pnasnet.model import NetworkImageNet
    from pnasnet.genotypes import PNASNet

    num_classes = 8

    model = NetworkImageNet(num_conv_filters, num_classes, num_cells, False, PNASNet)
    model.drop_path_prob = 0

    return model


emotion_models = {'resnet18': resnet_18, 'pnasnet': pnasnet}

