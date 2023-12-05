import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
my_cmap = matplotlib.colormaps['tab10']


os.makedirs('./img',exist_ok=True)
plt.figure(figsize=(7,5))
plt.title('Avg test accuracy on CIFAR-10')
plt.grid(True)
x = np.array([0, 5, 10, 15, 20]) / 20
acc = np.array([48, 50, 52, 50, 40])
std = np.array([0.92, 1.08, 0.57, 0.68, 1.09])
plt.xlabel(r'$\epsilon_{\mathrm{lp}}/\epsilon$' + ' Fraction of Total Privacy Budget Alotted to LP')
plt.plot([0, 0.2, 0.38, 0.6, 0.8, 1], [69, 72.2, 72, 71.2, 68.7, 61.3], color=my_cmap(0), linestyle='--', label=r'WideResNet-40 (RP), $\epsilon=1$ (Tang et al., 2023)') # data from https://arxiv.org/abs/2306.06076
plt.axhline(y=84.4, color=my_cmap(1), linestyle='--', label=r'FT, NFNet-F3 (SP), $\epsilon=1$ (De et al., 2022)') # data from https://arxiv.org/abs/2204.13650
plt.axhline(y=61, color=my_cmap(2), linestyle='-', label=r'LP, ResNet-50 (MOCO), $\epsilon=0.5$') # data from CLIPcifar10 folder
plt.plot(x, acc, color=my_cmap(3), label=r'ResNet-18 (SP), $\epsilon=0.76$')
plt.fill_between(x, y1=acc-std, y2=acc+std, color=my_cmap(3), alpha=0.3)
plt.axhline(y=45, color=my_cmap(4), linestyle='--', label=r'CNN (SC), $\epsilon=1$')
plt.legend()
plt.tight_layout()
plt.savefig('img/resnet18eps0.76.png')