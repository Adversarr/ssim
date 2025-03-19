import matplotlib.pyplot as plt
import os
import numpy as np
num_files = len([f for f in os.listdir('.') if f.startswith('deform_')])

plt.figure()
# 3d
ax = plt.subplot(111, projection='3d')
for i in range(num_files):
    data = np.load(f'deform_{i}.npy')
    ax.cla()
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.pause(0.1)
