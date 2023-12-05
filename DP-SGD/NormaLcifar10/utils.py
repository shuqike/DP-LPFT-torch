import argparse
import datetime
import random
import numpy as np
import torch


def set_seeds(seed):
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%B%d_%H:%M'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    # refer to https://github.com/zhreshold/mxnet/blob/2fbfdbcbacff8b738bd9f44e9c8cefc84d6dfbb5/example/gluon/train_imagenet.py
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--gamma', '--g', default=-1, type=float,
                        help='exponential decay of learning rate')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--save-weights', action='store_true')
    args = parser.parse_args()
    return args


def get_2phase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%B%d_%H:%M'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--eps-frac', type=float, default=0.5)
    parser.add_argument(
        "--lpbs",
        type=int,
        default=810,
        help="lp batch size",
    )
    parser.add_argument(
        "--ftbs",
        type=int,
        default=100,
        help="lp batch size",
    )
    parser.add_argument('--lplr', default=5e-5, type=float, help='initial linear probing learning rate')
    parser.add_argument('--ftlr', default=5e-5, type=float, help='initial finetuning learning rate')
    parser.add_argument('--lpgamma', default=-1, type=float,
                        help='exponential decay of lp learning rate')
    parser.add_argument('--ftgamma', default=-1, type=float,
                        help='exponential decay of ft learning rate')
    parser.add_argument('--multi-gpu', action='store_true')
    args = parser.parse_args()
    return args