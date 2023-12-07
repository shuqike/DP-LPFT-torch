import os
import time
import pickle
import torch
import torch.nn as nn
from fastDP import PrivacyEngine
from models.imnet_resnet import ResNet50CIFAR10
from utils import get_args, get_dataloaders, setup_seeds, freeze_bottom, unfreeze_all, test_model
from test.basics import debug_ResNet50, debug_unfreeze, debug_freeze, debug_img


def linear_prob(args, model, train_loader, test_loader):
    freeze_bottom(model)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lplr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=64,
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
            optimizer.step()
            optimizer.zero_grad()
        test_acc, test_loss = test_model(args, model, test_loader)
        args.log['test_acc'].append(test_acc)
        args.log['test_loss'].append(test_loss)
    privacy_engine.detach()


def finetune(args, model, train_loader, test_loader):
    unfreeze_all(model)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ftlr)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=64,
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
            optimizer.step()
            optimizer.zero_grad()
        test_acc, test_loss = test_model(args, model, test_loader)
        args.log['test_acc'].append(test_acc)
        args.log['test_loss'].append(test_loss)
    privacy_engine.detach()


def run(args):
    setup_seeds(args)
    train_loader, test_loader = get_dataloaders(args)
    model = ResNet50CIFAR10(args)
    args.log = {'test_acc': [], 'test_loss': []}
    start_time = time.time()

    linear_prob(args, model, train_loader, test_loader)
    finetune(args, model, train_loader, test_loader)

    args.time_spent = time.time() - start_time
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./temp', exist_ok=True)
    # save logs
    with open(f'logs/{args.name}.pkl', 'wb') as file:
        pickle.dump(args, file)
    # save model
    torch.save(model.state_dict(), f'temp/{args.name}.pt')


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        debug_ResNet50(args)
        debug_img()
        model = debug_freeze(ResNet50CIFAR10(args))
        debug_unfreeze(model)
        exit()
    run(args)
