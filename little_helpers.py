import numpy as np


def y_at_x(x, x_values, y_values):
    """
    Find values in y_values that belongs to the values in x_values clostest to x.

    Parameters
    ----------
    x : float or list of float
        The values to look for in x_values. If the exact values do not exist,
        the values in x_values clostest to x will be used.
    x_values : ndarray
        The array containing the independent variable.
    y_values : ndarray
        The array containing the dependent variable.

    Returns
    -------
    ndarray
        The values in y_values that are at the indices of the values of
        x_values that are clostest to the values in x.

    """
    return y_values[closest_index(x, x_values)]

def closest_index(x, x_values):
    """
    Find the index of a value in array x_values that is clostest to x.

    Parameters
    ----------
    x : float or list of float
        A single value or a list of values to look for in x_values.
    x_values : ndaaray
        The array with the values to compare to x.

    Returns
    -------
    ndarray
        The indices of the values in x_values that are clostest to x.

    """
    return np.argmin(np.abs(x-x_values[:, np.newaxis]), axis=0)