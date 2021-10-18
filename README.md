[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://app.travis-ci.com/AlexanderSouthan/little_helpers.svg?branch=main)](https://app.travis-ci.com/AlexanderSouthan/little_helpers)
[![codecov](https://codecov.io/gh/AlexanderSouthan/little_helpers/branch/main/graph/badge.svg?token=W7O1I2YKGO)](https://codecov.io/gh/AlexanderSouthan/little_helpers)

# little_helpers
Some helpful functions that keep being used in various of my repositories, install via:
```
pip install little_helpers
```

## array_tools.py
* y_at_y: Find values in y_values that belongs to the values in x_values clostest to x.
* closest_index: Find the index of a value in array x_values that is clostest to x.
* segment_xy_values: Segment the x_values and y_values according to segment borders.

## math_functions.py
* langmuir_isotherm: Calculate the q_e values of a simple Langmuir isotherm.
* langmuir_isotherm_hydrogel: Calculate the adsotpion inside a hydrogel based on a Langmuir model taking the swelling into account.
* langmuir_comp: Calculate the q_e values of a Langmuir isotherm taking into account competetive adsorption of two species.
* triangle: Calculate a triangle function.
* gaussian: Calculate one or a superposition of Gaussian normal distributions.
* boxcar: Calculate a boxcar function.
* boxcar_convolution: Calculate the convolution of a boxcar function with another function.
* piecewise_polynomial: Calculate the y values of a piecewise polynomial.
* flory_rehner: Calculate 1/M_c according to the Flory-Rehner equation.
* Herschel_Bulkley: Calculate the stress according tot he Herschel-Bulkley model.
* cum_dist_normal: Cumulative distribution function for the normal distribution.
* cum_dist_normal_with_rise: Superposition of cum_dist_normal and a linear function through the origin.

## num_derive.py
A simple method to calculate the derivative of discrete data.

## statsmodel_wrapper.py
A universal sklearn-style wrapper for statsmodels regressors.
