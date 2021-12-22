#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.little_helpers import array_tools


class TestArrayTools(unittest.TestCase):

    def test_array_tools(self):

        # check for robustness to unsorted list input
        x_list = [5, 8, 9, 88, 56, 22]
        y_list = [33, 99, 111, 2222, 8, 1]
        idx = array_tools.closest_index(68, x_list)
        y_at_x = array_tools.y_at_x(7, x_list, y_list)

        segs = array_tools.segment_xy_values(x_list, [23, 86])

        self.assertEqual(idx, 4)
        self.assertEqual(y_at_x, 99)
        self.assertEqual(len(segs), 3)
        self.assertEqual(len(segs[0]), 4)
        self.assertEqual(len(segs[1]), 3)
        self.assertEqual(len(segs[2]), 1)

        # test with ascending ndarray input
        sort_idx = np.argsort(x_list)
        x_array = np.array(x_list)[sort_idx]
        y_array = np.array(y_list)[sort_idx]

        idx = array_tools.closest_index(68, x_array)
        y_at_x = array_tools.y_at_x(7, x_array, y_array)

        segs = array_tools.segment_xy_values(x_array, [23, 86])

        self.assertEqual(idx, 4)
        self.assertEqual(y_at_x, 99)
        self.assertEqual(len(segs), 3)
        self.assertEqual(len(segs[0]), 4)
        self.assertEqual(len(segs[1]), 3)
        self.assertEqual(len(segs[2]), 1)

        # test with descending ndarray input
        x_array = np.array(x_list)[sort_idx[::-1]]
        y_array = np.array(y_list)[sort_idx[::-1]]

        segs = array_tools.segment_xy_values(x_array, [23, 86])

        self.assertEqual(len(segs), 3)
        self.assertEqual(len(segs[0]), 4)
        self.assertEqual(len(segs[1]), 3)
        self.assertEqual(len(segs[2]), 1)

        # test segmentation with x and y values
        segs_x, segs_y = array_tools.segment_xy_values(
            x_array, [23, 86], y_values=y_array)
