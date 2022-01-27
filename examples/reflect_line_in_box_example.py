# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:19:51 2022

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from little_helpers.geometry import reflect_line_in_box


# There seems to be a problem with the reflect function when it is out of the
# pyRandomWalk package. This is not resolved currently.
box = {'x': [-0.3, 0.5], 'y':[-1.2, 1.5]}


p1 = np.array([[0, 0]])
p2 = np.array([[3.3, -3]])

points_on_box, final_points = reflect_line_in_box(
    p1, p2, box=box)

fig1, ax1 = plt.subplots(dpi=600)
ax1.plot(points_on_box[0][0], points_on_box[0][1], ls='-', c='k')
ax1.plot([p1[0, 0], points_on_box[0][0, 0]],
         [p1[0, 1], points_on_box[0][1, 0]], ls='-', c='k')
ax1.plot([points_on_box[0][0, -1], final_points[0, 0]],
          [points_on_box[0][1, -1], final_points[0, 1]], ls='-', c='k')
ax1.plot(p1[0, 0], p1[0, 1], ls='none', marker='o', c='b')

ax1.plot(final_points[0, 0], final_points[0, 1], ls='none', marker='o', c='orange')
ax1.plot(points_on_box[0][0], points_on_box[0][1], ls='none', marker='o', c='r')

box = patches.Rectangle((box['x'][0], box['y'][0]),
                        box['x'][1]-box['x'][0],
                        box['y'][1]-box['y'][0],
                        linewidth=1, edgecolor='k', facecolor='none', ls='--')
ax1.add_patch(box)
ax1.set_aspect('equal', adjustable='box')
