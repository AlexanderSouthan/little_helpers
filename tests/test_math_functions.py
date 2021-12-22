#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.little_helpers import math_functions


class TestMathFunctions(unittest.TestCase):

    def test_math_functions(self):

        c_e = [1, 2, 3, 4]
        q_m = 3
        K_s = 0.2
        y_1 = math_functions.langmuir_isotherm(c_e, q_m, K_s)

        phi_h2o = 0.8
        y_2 = math_functions.langmuir_isotherm_hydrogel(c_e, q_m, K_s, phi_h2o)

        c_e_2 = [5, 6, 7, 8]
        K_s_2 = 0.8
        y_3 = math_functions.langmuir_comp(c_e, c_e_2, q_m, K_s, K_s_2)

        x = list(range(50))
        y_4 = math_functions.triangle(x, 22.3, 44.2, 33.1, 12.05, 3)
        y_5 = math_functions.gaussian(
            x, [1.2, 5], [3, 33.5], [1.1, 1], [4, 0.3])
        y_6 = math_functions.boxcar(x, 12.1, 15.3)
        y_7 = math_functions.boxcar_convolution(
            x, 12.3, 35.25, math_functions.gaussian, [1, 8.1, 0, 2],
            y_offset=1.1)
        y_8 = math_functions.piecewise_polynomial(
            x, [[3.2, 4.1, 5.1], [2, 4.1]], 33.2)
        y_9 = math_functions.piecewise_polynomial(
            x, [3.2, 4.1, 5.1], [])
        y_10 = math_functions.flory_rehner(
            [0.1, 0.2], 7000.3, 0.2, 0.4)
        y_11 = math_functions.Herschel_Bulkley(x, 200, 3, 0.6)
        y_12 = math_functions.cum_dist_normal(x, 3, 12.1, amp=3.2)
        y_13 = math_functions.cum_dist_normal_with_rise(
            x, 3.2, 25.8, 2.01, amp=1.1, linear_rise='full')
        y_14 = math_functions.cum_dist_normal_with_rise(
            x, 3.2, 25.8, 2.01, amp=1.1, linear_rise='left')
        y_15 = math_functions.cum_dist_normal_with_rise(
            x, 3.2, 25.8, 2.01, amp=1.1, linear_rise='right')
