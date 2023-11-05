# ResNet for [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1)

## Preparation

1. Randomly divide the 2000 datapoints into a training dataset of 1500 images and a test dataset of 500 images.

2. Use pretrained models on CIFAR-10, ResNet-18 for example:
    ```python
    import timm
    model = timm.create_model("resnet18_cifar10", pretrained=True)
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
