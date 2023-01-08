#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:47:02 2022

@author: almami
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

from little_helpers.geometry import (
    point_inside_circle, point_inside_cartesianbox, point_inside_polygon)


# Calculate random points
x_values = np.random.random(200)
y_values = np.random.random(200)

# Define a circle
x_c = 0.6
y_c = 0.7
r = 0.2
# Find point inside circle
inside_circle_mask = point_inside_circle(
    x_values, y_values, x_c=x_c, y_c=y_c, r=r)

# Define a rectangle
x_limits = [0.1, 0.3]
y_limits = [0.2, 0.6]
# Find points inside rectangle
inside_rect_mask = point_inside_cartesianbox(
    x_values, y_values, x_limits=x_limits, y_limits=y_limits)

# Define a polygon
polygon_x = [0.4, 0.6, 0.65, 0.7, 0.9, 0.72, 0.8, 0.65, 0.6, 0.6]
polygon_y = [0.2, 0.3, 0.5, 0.3, 0.2, 0.15, 0, 0.15, 0, 0.15]
# np.array(
#     [[0.4, 0.2], [0.6, 0.3], [0.65, 0.5], [0.7, 0.3], [0.9, 0.2], [0.72, 0.15],
#      [0.8, 0], [0.65, 0.15], [0.6, 0], [0.6, 0.15]])
# find points inside polygon
inside_polygon_mask = point_inside_polygon(
    x_values, y_values, polygon_x, polygon_y)

# Plot results
fig1, ax1 = plt.subplots()
# Plot points neither inside circle nor rectangle nor polygon
ax1.plot(x_values[~inside_circle_mask & ~inside_rect_mask & ~inside_polygon_mask],
         y_values[~inside_circle_mask & ~inside_rect_mask & ~inside_polygon_mask],
         ls='none', marker='o')
# Plot points inside circle
ax1.plot(x_values[inside_circle_mask], y_values[inside_circle_mask],
         ls='none', marker='o')
# Plot points inside rectangle
ax1.plot(x_values[inside_rect_mask], y_values[inside_rect_mask],
         ls='none', marker='o')
# Plot points inside polygon
ax1.plot(x_values[inside_polygon_mask], y_values[inside_polygon_mask],
         ls='none', marker='o')
# Plot circle
ax1.add_patch(
    Circle((x_c, y_c), radius=r, fill=False, edgecolor='k', ls='--'))
# Plot Rectangle
ax1.add_patch(
    Rectangle((x_limits[0], y_limits[0]), x_limits[1]-x_limits[0],
              y_limits[1]-y_limits[0], fill=False, edgecolor='k', ls='--'))
# Plot polygon
ax1.add_patch(
    Polygon(np.array([polygon_x, polygon_y]).T, closed=True, fill=False,
            edgecolor='k', ls='--'))
