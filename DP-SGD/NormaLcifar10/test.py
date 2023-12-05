import torch
from fastDP import PrivacyEngine
from torchvision.models import resnet18
model = resnet18(weights='IMAGENET1K_V1')
print(model)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# privacy_engine = PrivacyEngine(
#     model,
#     batch_size=64,
#     sample_size=50000, # size of CIFAR10 training set
#     epochs=50,
#     target_epsilon=1,
#     clipping_fn='automatic',
#     clipping_mode='MixOpt',
#     origin_params=None,
#     clipping_style='all-layer',
# )
# privacy_engine.attach(optimizer)
# print(privacy_engine.get_privacy_spent())