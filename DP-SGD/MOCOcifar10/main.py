import time
import torch
import torch.nn as nn
from fastDP import PrivacyEngine
from models.imnet_resnet import ResNet50CIFAR10, debug_ResNet50
from utils import get_args, get_dataloaders, setup_seeds, freeze_bottom, debug_img, debug_freeze


def run(args):
    setup_seeds(args)
    train_loader, test_loader = get_dataloaders(args)
    model = ResNet50CIFAR10(args)
    # start linear probing
    freeze_bottom(model)


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        debug_ResNet50(args)
        debug_img()
        debug_freeze(ResNet50CIFAR10(args))
        exit()
    run(args)
