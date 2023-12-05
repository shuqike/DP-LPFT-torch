from torchvision.models import resnet18
model = resnet18(weights='IMAGENET1K_V1')
print(model)