# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:19:51 2022

@author: Alexander Southan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from little_helpers.geometry import reflect_line_in_box


# 2D box
box = {'x': [-0.9, 0.5], 'y':[-1.2, 1.5]}

p1 = np.array([[0, 0]])
p2 = np.array([[13, -9]])

points_on_box, final_points = reflect_line_in_box(p1, p2, limits=box)

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


# 3D box
p1 = np.array([[0.4, 3, -2]])
p2 = np.array([[8.6, -5.3, -1]])

limits = {'x': [-1, 1], 'y':[-1, 3], 'z':[-5, -1.6]}

re_box, p_f = reflect_line_in_box(p1, p2, limits=limits)

# Probe über Berechnung von Linienlänge zwischen Punkten vor und nach Spiegelung
lengths_start = np.sqrt(((p1-p2)**2).sum(axis=1))

all_points = [[np.concatenate([[p1[pp, ii]], re_box[pp][ii], [p_f[pp, ii]]]) for ii in range(len(p1[0]))] for pp in range(p1.shape[0])]
lengths_end = np.array([])
for curr_points in all_points:
    diffs = []
    for curr_dim in range(p1.shape[1]):
        diffs.append(np.diff(curr_points[curr_dim]))
    curr_length = 0
    for curr_diffs in diffs:
        curr_length += curr_diffs**2
    curr_length = np.sqrt(curr_length).sum()
    lengths_end = np.append(lengths_end, curr_length)


fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

point = 0
for pp in [point]:#range(p1.shape[0]):
    ax_3d.plot(*[np.concatenate([[p1[pp, ii]], re_box[pp][ii], [p_f[pp, ii]]]) for ii in range(len(p1[0]))], marker='o')
    ax_3d.scatter(*[[p1[pp, ii], p_f[pp, ii]] for ii in range(len(p1[0]))], c='r', s=150)
