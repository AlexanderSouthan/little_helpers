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

        input_array = np.array([5, 8, 9, 88, 56, 22])
        idx = array_tools.closest_index(68, input_array)
        
        self.assertEqual(idx, 4)
