import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12.0, 7.0))
ax = plt.subplot(111)
t = np.arange(-3, 1, 0.01)
s = 1 / (1 + np.exp(-t))
line, = plt.plot(t, s, lw=2)
plt.ylim(0,1)
plt.xlabel('Coefficient of determination', fontsize=18)
plt.ylabel('Fitness', fontsize=16)
fig.savefig('./logistic_function.png')