#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.little_helpers import hydrogel_conversions


class TestHydrogelConversions(unittest.TestCase):

    def test_hydrogel_conversions(self):

        # test the different arguments
        eds_values = [1, 2, 3, 4]

        v_2s_1 = hydrogel_conversions.eds_to_volume_fraction(
            eds_values, eds_mode='plain', output='polymer')
        v_2s_2 = hydrogel_conversions.eds_to_volume_fraction(
            eds_values, eds_mode='subtracted', output='polymer')

        self.assertEqual((v_2s_2.sum()), np.array([0.5, 1/3, 0.25, 0.2]).sum())
        self.assertEqual((v_2s_1.sum()), np.array([1, 0.5, 1/3, 0.25]).sum())

        v_2s_3 = hydrogel_conversions.eds_to_volume_fraction(
            eds_values, eds_mode='plain', output='solvent')
        v_2s_4 = hydrogel_conversions.eds_to_volume_fraction(
            eds_values, eds_mode='subtracted', output='solvent')

        self.assertEqual((v_2s_4.sum()),
                         (1-np.array([0.5, 1/3, 0.25, 0.2])).sum())
        self.assertEqual((v_2s_3.sum()),
                         (1-np.array([1, 0.5, 1/3, 0.25])).sum())

        # test error messages
        self.assertRaises(
            ValueError, hydrogel_conversions.eds_to_volume_fraction,
            eds_values, 1, 1, 'subtacted', 'polymer')
        self.assertRaises(
            ValueError, hydrogel_conversions.eds_to_volume_fraction,
            eds_values, 1, 1, 'subtracted', 'polmer')
