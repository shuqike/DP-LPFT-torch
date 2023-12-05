import os
import numpy as np
import matplotlib.pyplot as plt


os.makedirs('./img',exist_ok=True)
plt.title(r'resnet-18, $\epsilon=0.76$')
x = np.array([0, 5, 10, 15, 20]) / 20
acc = np.array([43, 45, 52, 50, 37])
std = np.array([0.92, 1.08, 0.57, 0.68, 1.09])
plt.xlabel(r'$\epsilon_{\mathrm{lp}}/\epsilon$' + ' Fraction of Total Privacy Budget Alotted to LP')
plt.plot(x, acc)
plt.fill_between(x, y1=acc-std, y2=acc+std, alpha=0.3)
plt.savefig('img/resnet18eps0.76.png')