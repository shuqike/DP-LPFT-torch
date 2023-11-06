# Model example

Use google/mobilenet_v2_0.35_96 as an example.

## Summary of model

```
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
MobileNetV2ForImageClassification                  [1, 1001]                 --
├─MobileNetV2Model: 1-1                            [1, 1280]                 --
│    └─MobileNetV2Stem: 2-1                        [1, 8, 48, 48]            --
│    │    └─MobileNetV2ConvLayer: 3-1              [1, 16, 48, 48]           (464)
│    │    └─MobileNetV2ConvLayer: 3-2              [1, 16, 48, 48]           (176)
│    │    └─MobileNetV2ConvLayer: 3-3              [1, 8, 48, 48]            (144)
│    └─ModuleList: 2-2                             --                        --
│    │    └─MobileNetV2InvertedResidual: 3-4       [1, 8, 24, 24]            (1,408)
│    │    └─MobileNetV2InvertedResidual: 3-5       [1, 8, 24, 24]            (1,408)
│    │    └─MobileNetV2InvertedResidual: 3-6       [1, 16, 12, 12]           (1,808)
│    │    └─MobileNetV2InvertedResidual: 3-7       [1, 16, 12, 12]           (4,352)
│    │    └─MobileNetV2InvertedResidual: 3-8       [1, 16, 12, 12]           (4,352)
│    │    └─MobileNetV2InvertedResidual: 3-9       [1, 24, 6, 6]             (5,136)
│    │    └─MobileNetV2InvertedResidual: 3-10      [1, 24, 6, 6]             (8,832)
│    │    └─MobileNetV2InvertedResidual: 3-11      [1, 24, 6, 6]             (8,832)
│    │    └─MobileNetV2InvertedResidual: 3-12      [1, 24, 6, 6]             (8,832)
│    │    └─MobileNetV2InvertedResidual: 3-13      [1, 32, 6, 6]             (10,000)
│    │    └─MobileNetV2InvertedResidual: 3-14      [1, 32, 6, 6]             (14,848)
│    │    └─MobileNetV2InvertedResidual: 3-15      [1, 32, 6, 6]             (14,848)
│    │    └─MobileNetV2InvertedResidual: 3-16      [1, 56, 3, 3]             (19,504)
│    │    └─MobileNetV2InvertedResidual: 3-17      [1, 56, 3, 3]             (42,112)
│    │    └─MobileNetV2InvertedResidual: 3-18      [1, 56, 3, 3]             (42,112)
│    │    └─MobileNetV2InvertedResidual: 3-19      [1, 112, 3, 3]            (61,040)
│    └─MobileNetV2ConvLayer: 2-3                   [1, 1280, 3, 3]           --
│    │    └─Conv2d: 3-20                           [1, 1280, 3, 3]           (143,360)
│    │    └─BatchNorm2d: 3-21                      [1, 1280, 3, 3]           (2,560)
│    │    └─ReLU6: 3-22                            [1, 1280, 3, 3]           --
│    └─AdaptiveAvgPool2d: 2-4                      [1, 1280, 1, 1]           --
├─Dropout: 1-2                                     [1, 1280]                 --
├─Linear: 1-3                                      [1, 1001]                 1,282,281
====================================================================================================
Total params: 1,678,409
```

## Detailed model structure

```python
MobileNetV2ForImageClassification(
  (mobilenet_v2): MobileNetV2Model(
    (conv_stem): MobileNetV2Stem(
      (first_conv): MobileNetV2ConvLayer(
        (convolution): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (normalization): BatchNorm2d(16, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        (activation): ReLU6()
      )
      (conv_3x3): MobileNetV2ConvLayer(
        (convolution): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), groups=16, bias=False)
        (normalization): BatchNorm2d(16, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        (activation): ReLU6()
      )
      (reduce_1x1): MobileNetV2ConvLayer(
        (convolution): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (normalization): BatchNorm2d(8, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
      )
    )
    (layer): ModuleList(
      (0): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), groups=48, bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(8, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (1): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), groups=48, bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(8, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (2): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), groups=48, bias=False)
          (normalization): BatchNorm2d(48, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(48, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(16, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (3-4): 2 x MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(96, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), groups=96, bias=False)
          (normalization): BatchNorm2d(96, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(16, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (5): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(96, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), groups=96, bias=False)
          (normalization): BatchNorm2d(96, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(24, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (6-8): 3 x MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(144, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), groups=144, bias=False)
          (normalization): BatchNorm2d(144, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(24, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (9): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(144, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), groups=144, bias=False)
          (normalization): BatchNorm2d(144, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(32, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (10-11): 2 x MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(192, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False)
          (normalization): BatchNorm2d(192, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(32, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (12): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(192, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), groups=192, bias=False)
          (normalization): BatchNorm2d(192, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(56, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (13-14): 2 x MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(336, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), groups=336, bias=False)
          (normalization): BatchNorm2d(336, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(56, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
      (15): MobileNetV2InvertedResidual(
        (expand_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(336, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (conv_3x3): MobileNetV2ConvLayer(
          (convolution): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), groups=336, bias=False)
          (normalization): BatchNorm2d(336, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
          (activation): ReLU6()
        )
        (reduce_1x1): MobileNetV2ConvLayer(
          (convolution): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (normalization): BatchNorm2d(112, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
        )
      )
    )
    (conv_1x1): MobileNetV2ConvLayer(
      (convolution): Conv2d(112, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (normalization): BatchNorm2d(1280, eps=0.001, momentum=0.997, affine=True, track_running_stats=True)
      (activation): ReLU6()
    )
    (pooler): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (dropout): Dropout(p=0.2, inplace=True)
  (classifier): Linear(in_features=1280, out_features=1001, bias=True)
)
```
