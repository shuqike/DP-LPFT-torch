import os
import pickle
import random
import datetime
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from get_data import get_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime('%B%d_%H:%M'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft_type', type=str, choices=['lp', 'ft'])
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--sigma', type=float, required=True)
    # Delta: The target δ of the (ϵ,δ)-differential privacy guarantee.
    # Generally, it should be set to be less than the inverse of the size of the training dataset.
    # We set it to $10^{−5}$ as the CIFAR10 dataset has 50,000 training points.
    parser.add_argument('--delta', type=float, default=1e-5)
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
    args = parser.parse_args()
    return args


def priv_compatible(model):
    summary(model, input_size=(1, 3, 96, 96), dtypes=['torch.FloatTensor'])
    # Check if the model is compatible with the privacy engine
    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    assert errors == []
    return model


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def validate(model, device, valid_loader):
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


def run(args, state_path, model, train_loader, test_loader):
    run_results = []
    for run_id in range(args.n_runs):
        # Move the model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        one_run_result = []
        for epoch in range(args.max_epochs):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            test_loss, test_acc = validate(model, device, test_loader)
            one_run_result.append(test_loss, test_acc)
        run_results.append(one_run_result)


if __name__ == '__main__':
    args = get_args()
    os.makedirs('temp', exist_ok=True)
    state_path = os.path.join('temp/', args.name)
    os.makedirs(state_path, exist_ok=True)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_dataloader(args.batch_size)

    # Initialize model
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
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
        # Move the model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    else:
        raise NotImplementedError

    run(args, state_path, model, train_loader, test_loader)