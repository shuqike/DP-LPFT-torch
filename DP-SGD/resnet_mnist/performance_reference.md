# Performance reference

Here we check the previous SOTA performance and the successful hyper-parameter settings of DP-SGD on MNIST classification.

Papers:
- [DP-SGD vs PATE: Which Has Less Disparate Impact on Model Accuracy?](https://arxiv.org/pdf/2106.12576.pdf). For $\epsilon\in[0.5,15]$, the accuracy is around 90%~99% for class $0,1,2,3,4,5,6,7,9$ and around 20%~30% for class $8$. The difference is due to the severe disparate impact of classification subgroups using DP-SGD. They used ResNet-18 and learning rate 0.01 for DP-SGD.
