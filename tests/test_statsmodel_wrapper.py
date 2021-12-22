#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import statsmodels.api as sm
import unittest

from src.little_helpers import statsmodel_wrapper


class TestSatsmodelWrapper(unittest.TestCase):

    def test_statsmodel_wrapper(self):
        x = np.arange(50)
        y = 3*x

        a = statsmodel_wrapper(sm.OLS)
        a.fit(x, y)
        pred = a.predict(x)

        for curr_y, curr_pred in zip(y, pred):
            self.assertAlmostEqual(curr_y, curr_pred, 5)
