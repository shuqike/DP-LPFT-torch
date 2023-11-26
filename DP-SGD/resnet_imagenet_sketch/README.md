# ResNet for ImageNet-Sketch

## Preparation

1. Download dataset

```python
from datasets import load_dataset
dataset = load_dataset('imagenet_sketch', data_dir='./data/')
```

## Single-phase fine-tuning

Study how the pretrained ResNet models (resnet18, resnet34, resnet50) perform when using

- non-private linear probing
- non-private full-finetuning
- dp linear probing
- dp full-finetuning

Our targeting experiment outcomes include:

- average convergence plots of test loss (mean_array, std_array)

- average convergence plots of training loss (mean_array, std_array)

- average early stopping test loss (mean_float, std_float)

## Two-phase fine-tuning

Study how the pretrained ResNet models (resnet18, resnet34, resnet50) perform when using

- non-private linear-probing-then-finetuning
- dp-linear-probing-then-finetuning
