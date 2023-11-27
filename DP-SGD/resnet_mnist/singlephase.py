# Refer to https://www.kaggle.com/code/paulojunqueira/mnist-with-pytorch-and-transfer-learning-timm
import os
import pickle
import random
import datetime
import argparse
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision
from torchinfo import summary
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from get_data import get_dataloader
import singlephase_utils as utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%B%d_%H:%M'))
    parser.add_argument('--ft_type', type=str, choices=['lp', 'ft'])
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--sigma', type=float, required=True)
    # Delta: The target δ of the (ϵ,δ)-differential privacy guarantee.
    # Generally, it should be set to be less than the inverse of the size of the training dataset.
    # We set it to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--max_per_sample_grad_norm', type=float, default=1.0)
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--disable_dp', action='store_true')
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    # refer to https://github.com/zhreshold/mxnet/blob/2fbfdbcbacff8b738bd9f44e9c8cefc84d6dfbb5/example/gluon/train_imagenet.py
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    return args


def priv_compatible(model):
    summary(model, input_size=(1, 1, 96, 96), dtypes=['torch.FloatTensor'])
    # Check if the model is compatible with the privacy engine
    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    assert errors == []
    return model


def train(args, model, device, train_loader, optimizer, privacy_engine) -> List:
    criterion = nn.CrossEntropyLoss()
    losses = []
    step_inc = 0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        step_inc += 1
        if step_inc >= args.max_steps:
            break

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        return [np.mean(losses), epsilon], step_inc
    else:
        return [np.mean(losses)], step_inc


def test(model, device, valid_loader):
    with torch.inference_mode():
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(valid_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(valid_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(valid_loader.dataset),
                100.0 * correct / len(valid_loader.dataset),
            )
        )
        return test_loss, correct / len(valid_loader.dataset)


def run(args, save_timer, run_id, run_results, step, p_model, p_optimizer, p_train_loader, privacy_engine, test_loader):
    # Move the model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_model =p_model.to(device)

    one_run_result = []
    while step < args.max_steps:
        train_stat, step_inc = train(args, p_model, device, p_train_loader, p_optimizer, privacy_engine)
        step += step_inc
        test_loss, test_acc = test(p_model, device, test_loader)
        one_run_result += [train_stat, test_loss, test_acc]
        save_timer += 1
        if save_timer % args.save_freq == 0:
            utils.save(state_path, run_id, run_results, step, p_model, p_optimizer, privacy_engine, p_train_loader)
    run_results.append(one_run_result)
    with open(os.path.join(state_path, 'run_results.pkl'), 'wb') as file:
        pickle.dump(run_results, file)
    return save_timer, run_results


def run_all(args, state_path, model, train_loader, test_loader):
    save_timer = 0

    if args.resume:
        run_id, run_results, step, p_model, p_optimizer, privacy_engine, p_train_loader = utils.load_checkpoint(os.path.join(state_path, "checkpoint.pth.tar"))

    else:
        run_id = 0
        run_results = []
        step = 0
        # Set random seed
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        
        privacy_engine = None
        if not args.disable_dp:
            privacy_engine = PrivacyEngine()
            p_model, p_optimizer, p_train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
        else:
            p_model, p_optimizer, p_train_loader = model, optimizer, train_loader

    while run_id < args.n_runs:
        save_timer, run_results = run(args, save_timer, run_id, run_results, step, p_model, p_optimizer, p_train_loader, privacy_engine, test_loader)

        # Prepare for the next iteration
        run_id += 1
        step = 0
        # Set random seed
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        privacy_engine = None
        if not args.disable_dp:
            privacy_engine = PrivacyEngine()
            p_model, p_optimizer, p_train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
        else:
            p_model, p_optimizer, p_train_loader = model, optimizer, train_loader


if __name__ == '__main__':
    args = get_args()
    os.makedirs('temp', exist_ok=True)
    state_path = os.path.join('temp/', args.name)
    os.makedirs(state_path, exist_ok=True)
    # Save the arguments for this experiment
    with open(os.path.join(state_path, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    train_loader, test_loader = get_dataloader(args.batch_size)

    # Initialize model
    if args.model == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True, in_chans=1)
    elif args.model == 'resnet34':
        model = timm.create_model('resnet34', pretrained=True, in_chans=1)
    elif args.model == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, in_chans=1)
    else:
        raise NotImplementedError
    
    # Resize and reinitialize the linear head
    model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)

    if args.ft_type == 'lp':
        # linear probing preparation
        print('Start linear probing...')
        model = model.train()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        # Sanity check of model setting
        model = priv_compatible(model)

    elif args.ft_type == 'ft':
        print('Start fine-tuning...')
        model = model.train()
        for p in model.parameters():
            p.requires_grad = True
        # Sanity check of model setting
        model = priv_compatible(model)

    else:
        raise NotImplementedError

    # Move the model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_all(args, state_path, model, train_loader, test_loader)
