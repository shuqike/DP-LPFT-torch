import sys
sys.path.append('../')
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchinfo
from models.imnet_resnet import ResNet50
from utils import unfreeze_all, freeze_bottom


def debug_ResNet50(args):
    model = ResNet50(args.weights)
    print(model)


def debug_img():
    print('\ndebug...')
    transform = transforms.Compose( # refer to tan/config/dataset_cifar.yml, tailored for moco(v2-3)+resnet50
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
        ]
    )
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    print(train_dataset[0][0].shape)


def debug_freeze(model):
    print('\ndebug...')
    torchinfo.summary(model=model, input_size=(1, 3, 224, 224))
    freeze_bottom(model)
    torchinfo.summary(model=model, input_size=(1, 3, 224, 224))
    return model


def debug_unfreeze(model):
    print('\ndebug...')
    torchinfo.summary(model=model, input_size=(1, 3, 224, 224))
    unfreeze_all(model)
    torchinfo.summary(model=model, input_size=(1, 3, 224, 224))
