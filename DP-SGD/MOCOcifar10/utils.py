import argparse
import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchinfo


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%B%d_%H:%M'))
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--weights', type=str, default='models/checkpoint/mocov3-rn50-1000ep.pth.tar',
                        choices=[
                            'models/checkpoint/mocov3-rn50-1000ep.pth.tar',
                            'models/checkpoint/mocov3-rn50-300ep.pth.tar',
                            'models/checkpoint/mocov3-rn50-100ep.pth.tar',
                            'models/checkpoint/mocov2_rn50_200ep.pth.tar',
                            'models/checkpoint/mocov2_rn50_800ep.pth.tar'
                        ]
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lp-epoch', type=int, default=200)
    parser.add_argument('--noise-multiplier', type=float, default=1)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.device == '':
        args.device = "cuda" if torch.cuda.is_available() else \
            ("mps" if torch.backends.mps.is_available() else "cpu")
    return args


def get_dataloaders(args):
    transform = transforms.Compose( # refer to tan/config/dataset_cifar.yml, tailored for moco(v2-3)+resnet50
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
        ]
    )
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def setup_seeds(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def freeze_bottom(model):
    total_params = 0
    trainable_params = 0
    for (k,v) in model.named_parameters():
        total_params += v.numel()
        v.requires_grad = False
    for (k,v) in model._model.fc.named_parameters():
        trainable_params += v.numel()
        v.requires_grad = True
    print(f"Total parameters {total_params}. Trainable parameters {trainable_params}.")


def debug_img():
    transform = transforms.Compose( # refer to tan/config/dataset_cifar.yml, tailored for moco(v2-3)+resnet50
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
        ]
    )
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    print(train_dataset)


def debug_freeze(model):
    torchinfo.summary(model=model, input_size=(1, 3, 224))
