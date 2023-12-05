'''Refer to
https://github.com/openai/CLIP
https://github.com/awslabs/fast-differential-privacy/tree/main
'''
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import get_2phase_args


os.makedirs('./temp', exist_ok=True)
os.makedirs('./temp/lpft', exist_ok=True)
device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu")
args = get_2phase_args()
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

model = resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
model = ModuleValidator.fix(model)
if args.multi_gpu:
    model= nn.DataParallel(model)
model = model.to(device)
# Freeze bottom layers for linear probing
total_params = 0
trainable_params = 0
for (k,v) in model.named_parameters():
    total_params += v.numel()
    v.requires_grad = False
for (k,v) in model.fc.named_parameters():
    trainable_params += v.numel()
    v.requires_grad = True
print(f"Total parameters {total_params}. Trainable parameters {trainable_params}.")
optimizer = torch.optim.SGD(model.parameters(), lr=args.lplr)
privacy_engine = PrivacyEngine()
p_model, p_optimizer, p_train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=args.noise_multiplier,
    max_grad_norm=1.0,
)
criterion = nn.CrossEntropyLoss()

best_test_acc = 0
for epoch in range(args.epoch):

    # turn to full finetuning
    if epoch == args.lp_epoch:
        model2 = resnet18(weights='IMAGENET1K_V1')
        model2.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        model2 = ModuleValidator.fix(model2)
        model2 = model2.to(device)
        model2.load_state_dict(p_model._module.state_dict())
        # unfreeze all parameter
        for (k,v) in model2.named_parameters():
            total_params += v.numel()
            v.requires_grad = True
        optimizer = torch.optim.SGD(model2.parameters(), lr=args.ftlr)
        p_model, p_optimizer, p_train_loader = privacy_engine.make_private(
            module=model2,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=1.0,
        )

    with BatchMemoryManager(
        data_loader=p_train_loader,
        max_physical_batch_size=100,
        optimizer=p_optimizer) as memory_safe_data_loader:
        for batch in memory_safe_data_loader:
            p_optimizer.zero_grad()
            imgs, labels = batch[0].to(device), batch[1].to(device)
            loss = criterion(p_model(imgs), labels)
            loss.backward()
            p_optimizer.step()

    # Test
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            # calculate outputs by running images through the network
            outputs = p_model(imgs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %, privacy cost{privacy_engine.get_epsilon(1/50000)}')
    best_test_acc = max(best_test_acc, 100 * correct // total)

if args.save_weights:
    torch.save({
                'args': args,
                'state_dict': p_model._module.state_dict(),
                'privacy_spent': privacy_engine.get_epsilon(1/50000),
                'best_test_acc': best_test_acc,
                },
                f'temp/lpft/s{args.seed}ech{epoch}lpech{args.lp_epoch}eps{args.noise_multiplier}b{args.batch_size}lplr{args.lplr}ftlr{args.ftlr}.pt')
