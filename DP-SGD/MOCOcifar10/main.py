import time
import torch
import torch.nn as nn
from fastDP import PrivacyEngine
from models.imnet_resnet import ResNet50CIFAR10
from utils import get_args, get_dataloaders, setup_seeds, freeze_bottom, unfreeze_all, test
from test.basics import debug_ResNet50, debug_unfreeze, debug_freeze, debug_img


def linear_prob(args, model, train_loader, test_loader):
    freeze_bottom(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lplr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=50000, # size of CIFAR-10 training set
        epochs=args.lp_epoch,
        noise_multiplier=args.noise_multiplier,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
    )
    privacy_engine.attach(optimizer)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.lp_epoch):
        for i, batch in enumerate(train_loader):
            imgs, labels = batch[0].to(args.device), batch[1].to(args.device)
            loss = criterion(model(imgs), labels)
            loss.backward()
            if (i+1) % (args.batch_size/64) == 0:
                optimizer.step()
                optimizer.zero_grad()
        test(args, model, test_loader)
    privacy_engine.detach(optimizer)


def finetune(args, model, train_loader, test_loader):
    unfreeze_all(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ftlr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=50000, # size of CIFAR-10 training set
        epochs=args.epoch-args.lp_epoch,
        noise_multiplier=args.noise_multiplier,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
    )
    privacy_engine.attach(optimizer)
    criterion = nn.CrossEntropyLoss()
    for i in range(args.epoch - args.lp_epoch):
        for i, batch in enumerate(train_loader):
            imgs, labels = batch[0].to(args.device), batch[1].to(args.device)
            loss = criterion(model(imgs), labels)
            loss.backward()
            if (i+1) % (args.batch_size/64) == 0:
                optimizer.step()
                optimizer.zero_grad()
        test(args, model, test_loader)
    privacy_engine.detach(optimizer)


def run(args):
    setup_seeds(args)
    train_loader, test_loader = get_dataloaders(args)
    model = ResNet50CIFAR10(args)
    linear_prob(args, model, train_loader, test_loader)
    finetune(args, model, train_loader, test_loader)


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        debug_ResNet50(args)
        debug_img()
        model = debug_freeze(ResNet50CIFAR10(args))
        debug_unfreeze(model)
        exit()
    run(args)
