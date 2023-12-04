'''Refer to
https://github.com/openai/CLIP
https://github.com/awslabs/fast-differential-privacy/tree/main
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from fastDP import PrivacyEngine
from imnet_resnet import ResNet50CIFAR10
from utils import get_args


device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu")
args = get_args()
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
model = ResNet50CIFAR10()
if args.multi_gpu:
    model= nn.DataParallel(model)
model = model.to(device)
# Freeze bottom layers for linear probing
total_params = 0
trainable_params = 0
for (k,v) in model.named_parameters():
    total_params += v.numel()
    v.requires_grad = False
for (k,v) in model._model.fc.named_parameters():
    trainable_params += v.numel()
    v.requires_grad = True
print(f"Total parameters {total_params}. Trainable parameters {trainable_params}.")
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
privacy_engine = PrivacyEngine(
    model,
    batch_size=args.batch_size,
    sample_size=50000, # size of CIFAR10 training set
    epochs=args.epoch,
    target_epsilon=args.epsilon,
    clipping_fn='automatic',
    clipping_mode='MixOpt',
    origin_params=None,
    clipping_style='all-layer',
)
privacy_engine.attach(optimizer)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epoch):
    running_loss = 0.0
    for i, batch in enumerate(train_loader, 0):
        imgs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    # Test
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(imgs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
