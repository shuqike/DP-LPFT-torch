import os
import numpy as np
import matplotlib.pyplot as plt


os.makedirs('./img',exist_ok=True)
plt.title(r'resnet-18, $\epsilon=0.76$')
x = np.array([0, 5, 10, 15, 20]) / 20
acc = np.array([28, 39, 40, 40, 37])
plt.xlabel(r'$\epsilon_{\mathrm{lp}}/\epsilon$' + ' Fraction of Total Privacy Budget Alotted to LP')
plt.plot(x, acc)
plt.savefig('img/resnet18eps0.76.png')