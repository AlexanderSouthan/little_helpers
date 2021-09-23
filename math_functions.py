# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate

from .array_tools import segment_xy_values

def langmuir_isotherm(c_e, q_m, K_s):
    """
    Calculate the q_e values of a Langmuir isotherm.

    Parameters
    ----------
    c_e : ndarray
        The equilibrium concentrations in the liquid phase. Can have any shape,
        so an (M, N) array may be interpreted as M data rows with N data
        points.
    q_m : float
        The adsorption capacity of the adsorber.
    K_s : float
        The equilibrium constant of adsorption and desorption.

    Returns
    -------
    ndarray
        The equilibrium concentrations q_e in the adsorber. Has the same shape
        like c_e.

    """
    return q_m * c_e * K_s/(1 + c_e * K_s)

def langmuir_isotherm_hydrogel(c_e, q_m, K_s, phi_h2o, rho_hydrogel=1):
    """
    Calculate the a_e values of a Langmuir isotherm in a hydrogel.

    Parameters
    ----------
    c_e : TYPE
        The equilibrium concentrations in the liquid phase. Can have any shape,
        so an (M, N) array may be interpreted as M data rows with N data
        points. It is assumed that the aqueous phase within the hydrogel has
        the same concentration like c_e.
    q_m : TYPE
        The adsorption capacity of the polymer network within the hydrogel.
    K_s : TYPE
        The equilibrium constant of adsorption and desorption.
    phi_h2o : TYPE
        The volume fraction of water inside the hydrogel.
    rho_hydrogel : TYPE, optional
        The density of the hydrogel in g/mL. The default is 1.

    Returns
    -------
        The equilibrium concentrations a_e in the adsorber. Has the same shape
        like c_e.

    """
    return c_e*phi_h2o/rho_hydrogel + q_m * c_e * K_s/(1 + c_e * K_s)

def langmuir_comp(c_e_1, c_e_2, q_m, K_s_1, K_s_2):
    """
    Calculate the q_e values of a Langmuir isotherm for competitive adsorption.

    Parameters
    ----------
    c_e_1 : ndarray
        The equilibrium concentrations of adsorbat 1 in the liquid phase. Can 
        have any shape,so an (M, N) array may be interpreted as M data rows
        with N data points.
    c_e_2 : ndarray
        The equilibrium concentrations of adsorbat 2 in the liquid phase. Can 
        have any shape,so an (M, N) array may be interpreted as M data rows
        with N data points.
    q_m : float
        The adsorption capacity of the adsorber.
    K_s_1 : float
        The equilibrium constant of adsorption and desorption of adsorbat 1.
    K_s_2 : float
        The equilibrium constant of adsorption and desorption of adsorbat 2.

    Returns
    -------
    ndarray
        The equilibrium concentrations q_e in the adsorber. Has the same shape
        like c_e.

    """
    return q_m * c_e_1 * K_s_1/(1 + c_e_1 * K_s_1 + c_e_2 * K_s_2)

def triangle(x, start_left, start_right, x_max, y_max, y_offset=0):
    """
    Calculate a triangle function. 
    
    The triangle function is zero outside of the triangle and different slopes
    on both sides of the triangle are possible.

    Parameters
    ----------
    x : ndarray
        The x values used for the calculation. Can be any shape, but the
        triangle will always be produced in the last dimension. An (M, N) array
        can therefore be interpreted as M data rows with potentially different
        x values while the triangles are always created at the same x values.
    start_left : float
        The x value where the triangle starts, i.e. the left edge of the
        tiangle.
    start_right : float
        The x value where the triangle stops, i.e. the right edge of the
        triangle.
    x_max : float
        The x value of the triangle maximum/minimum, must be between start_left
        and start_right, otherwise odd results will occur.
    y_max : float
        The y value of the triangle maximum/minimum.
    y_offset : float, optional
        The y value outside of the triangle. Default is 0. If y_offset is
        greater than y_max, the triangle will point downwards, otherwise
        upwards.

    Returns
    -------
    triangle : ndarray
        An array containing the funtion values of the triangle functions. Has
        the same shape like x.

    """
    left_mask = np.logical_and(
        x >= min(start_left, x_max),
        x <= max(start_left, x_max))
    right_mask = np.logical_and(
        x > min(x_max, start_right),
        x <= max(x_max, start_right))

    left_slope = (y_max-y_offset)/(x_max - start_left)
    right_slope = -(y_max-y_offset)/(start_right - x_max)

    triangle = np.full_like(x, y_offset)
    triangle[left_mask] += left_slope * (x[left_mask] - start_left)
    triangle[right_mask] += right_slope * (x[right_mask] - start_right)
    return triangle

def gaussian(x, amp, x_offset, y_offset, sigma):
    """
    Calculate one or a superposition of Gaussian normal distributions.

    Parameters
    ----------
    x : ndarray
        A one-dimensional array with the x values used for calculations.
    amp : float or list of float
        The amplitudes, i.e. the maximum values of the calculated Gauss curve.
        If a single value is given, a single peak is created. If a list of
        values is given, a superposition of several Gauss curves will be
        calculated.
    x_offset : float or list of float
        The x position of the maximum value defined by amp. Must be the same
        shape like amp.
    y_offset : float or list of float
        The y value of the baseline of the Gauss curve. Must be the same shape
        like amp.
    sigma : float or list of float
        The with of the Gauss curve. The full width at half maximum is given by
        2*sqrt(2*ln(2))*sigma. Must be the same shape like amp.

    Returns
    -------
    ndarray
        An array containing the function values of the (superimposed) Gauss
        curves. Has the same shape like x.

    """
    amp = np.array(amp, ndmin=1)
    x_offset = np.array(x_offset, ndmin=1)
    y_offset = np.array(y_offset, ndmin=1)
    sigma = np.array(sigma, ndmin=1)
    return np.sum(
        amp[:, np.newaxis] * np.exp(
            (x - x_offset[:, np.newaxis])**2 /
            (-2 * sigma[:, np.newaxis]**2)) +
        y_offset[:, np.newaxis], axis=0)

def boxcar(x, boxcar_start, boxcar_end, y_offset=0, amp=1):
    boxcar_mask = np.logical_and(
        x >= min(boxcar_start, boxcar_end),
        x <= max(boxcar_start, boxcar_end))
    y_boxcar = np.full_like(x, y_offset)
    y_boxcar[boxcar_mask] = amp
    return y_boxcar

def boxcar_convolution(x, boxcar_start, boxcar_end, convolution_function,
                       con_func_params, y_offset=0):
    x_spacing = np.abs(x[1]-x[0])
    x_min = x[0]
    x_max = x[-1]
    boxcar_width = abs(boxcar_start - boxcar_end)

    x_addition_datapoints = np.around(
        boxcar_width/(2*x_spacing)).astype(np.uint32)
    x_addition = x_addition_datapoints * x_spacing
    x_min_convolution = x_min - x_addition
    x_max_convolution = x_max + x_addition

    x_values_convolution = np.arange(
        x_min_convolution, x_max_convolution+x_spacing/2, x_spacing)

    y_con_func = convolution_function(x_values_convolution, *con_func_params)
    y_con_func_integral = integrate.cumtrapz(y_con_func, x_values_convolution,
                                             initial=0)

    function_values = (y_con_func_integral[2*x_addition_datapoints:] -
                       y_con_func_integral[
                           :len(x_values_convolution) -
                           2*x_addition_datapoints] +
                       y_offset)
    return function_values

def piecewise_polynomial(x_values, coefs, segment_borders=[]):
    """
    Calculate the y values of a piecewise polynomial.

    Can also calculate a simple polynomial.

    Parameters
    ----------
    x_values : ndarray
        A 1D array with the length M holding the independent varibale used for
        calculation of the piecewise polynomial.
    coefs : list of ndarray
        A list containing the coefficient vectors of the polynomial equations
        for the data segments. Each list entry must be in a format so that it
        can be passed directly to np.polynomial.polynomial.polyval to calculate
        the polynomial values. If segment borders is left at the default value,
        still a list with only one coefficient vector must be given.
    segment_borders : list of int or float, optional
        The values with respect to x_values at which the data is divided into
        segments. An arbitrary number of segment borders may be given, but it
        is recommended to provide a sorted list in order to avoid confusion.
        If the list is not sorted, it will be sorted. The default is [] meaning
        that only a simple polynomial is calculated.

    Returns
    -------
    ndarray
        The y values of the piecewise polynomial, an array with the same length
        as x_values.

    """

    if segment_borders:
        x_segments = segment_xy_values(x_values, segment_borders)
    else:
        x_segments = [x_values]

    curve_segments = []
    for curr_x, curr_coefs in zip(x_segments, coefs):
        poly_vals = np.polynomial.polynomial.polyval(curr_x, curr_coefs)
        curve_segments.append(poly_vals
                              if len(curve_segments) == len(x_segments)-1
                              else poly_vals[:-1])

    return np.concatenate(curve_segments)