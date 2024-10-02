import torch
from torchvision.models import vit_b_16, vgg16_bn, resnet50
import torch.nn as nn
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_model(model, pretrained, num_classes):
    if model == 'resnet':
        net = resnet50(pretrained=pretrained)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    elif model == 'vgg':
        net = vgg16_bn(pretrained=pretrained)
        in_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features, num_classes) 
    else:
        net = vit_b_16(pretrained=pretrained)
        net.heads.head = torch.nn.Linear(in_features=net.heads.head.in_features, out_features=num_classes)

    return net

