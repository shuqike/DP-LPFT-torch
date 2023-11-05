# ResNet for ImageNet-Sketch

## Preparation

1. Download dataset

```bash
curl -X GET \
     "https://datasets-server.huggingface.co/first-rows?dataset=imagenet_sketch&config=default&split=train"
```

## Single-phase fine-tuning

Study how the pretrained ResNet models (resnet18, resnet34, resnet50) perform when using

- dp linear probing
- dp full-finetuning

Our targeting experiment outcomes include:

- average convergence plots of test loss (mean_array, std_array)

- average convergence plots of training loss (mean_array, std_array)

- average early stopping test loss (mean_float, std_float)

## Two-phase fine-tuning

Study how the pretrained ResNet models (resnet18, resnet34, resnet50) perform when using

- linear-probing-then-finetuning
