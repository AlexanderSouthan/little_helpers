#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.little_helpers import num_derive


class TestNumDerive(unittest.TestCase):

    def test_num_derive(self):
        x = list(range(50))
        y = []
        for curr_x in x:
            y.append(curr_x**2 + 0.31*curr_x - 3.2)
        y = [y]

        deriv_1 = num_derive.derivative(x, y, order=1, averaging_window=1)
        deriv_2 = num_derive.derivative(x, y, order=2, averaging_window=5)
