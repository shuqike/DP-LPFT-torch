# MobileNets for CIFAR-10

## Preparation

1. Dataset: CIFAR-10

2. Use features pretrained on ImageNet

## Single-phase fine-tuning

### Target

Study how the pretrained MobileNetsV2 (dimension control: width multiplier) perform when using

- non-private linear probing
- non-private full-finetuning
- dp linear probing
- dp full-finetuning

Our targeting experiment outcomes include:

- average convergence plots of test loss (mean_array, std_array)

- average convergence plots of training loss (mean_array, std_array)

- average early stopping test loss (mean_float, std_float)

### Outcomes

Linear probing parameter stat:

- google/mobilenet_v2_0.35_96
    - Total parameters count: 1678409
    - Trainable parameters count: 1282281
- google/mobilenet_v2_0.75_160
    - Total parameters count: 2637705
    - Trainable parameters count: 1282281
- google/mobilenet_v2_1.0_224
    - Total parameters count: 3506153
    - Trainable parameters count: 1282281
- google/mobilenet_v2_1.4_224
    - Total parameters count: 6110569
    - Trainable parameters count: 1794793

## Two-phase fine-tuning

Study how the pretrained MobileNetsV2 (dimension control: width multiplier) perform when using

- non-private linear-probing-then-finetuning
- dp-linear-probing-then-finetuning
