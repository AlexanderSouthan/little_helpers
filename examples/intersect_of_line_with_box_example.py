# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:19:51 2022

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from little_helpers.geometry import line_through_box

box = {'x': [-0.3, 0.5], 'y': [1.2, 1.5]}

x = np.array([-0.6, 3])
y = np.array([1.3, 2])

solution_score, overlap_x, overlap_y = line_through_box(
    x, y, box)

fig1, ax1 = plt.subplots(dpi=600)
ax1.plot(x, y)
ax1.plot(overlap_x, overlap_y)
box = patches.Rectangle((box['x'][0], box['y'][0]),
                        box['x'][1]-box['x'][0],
                        box['y'][1]-box['y'][0],
                        linewidth=1, edgecolor='k', facecolor='none', ls='--')
ax1.add_patch(box)
ax1.set_aspect('equal', adjustable='box')
