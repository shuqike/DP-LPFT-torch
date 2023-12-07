import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import hyperopt
from imnet_resnet import ResNet50CIFAR10