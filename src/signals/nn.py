# -*- coding: utf-8 -*-
"""
nn.py
Functions for manipulation of the NN signal.

@author: Arturo Moncada-Torres
"""
import numpy as np
from scipy.interpolate import CubicSpline

#%%
def interpolate(nn_x, nn_y, from_time, to_time, T_int=0.2):
    """ 
    Interpolate the NN-intervals to obtain a smooth, equidistant signal.
    
    Parameters
    ----------
    nn_x: numpy array
        Location (either in samples or time) of the NN intervals
    nn_y: numpy array
        Values of the NN intervals
    from_time: float
        Initial time (in seconds) of the interpolated signal
    to_time: float
        Final time (in seconds) of the interpolated signal
    T_int: float
        Interpolation interval. Defaults to 0.2 s, as recommended
        in the original paper (see References).
        
    Returns
    -------
    nn_interp_x: numpy array
        Location along the x-axis of the NN intervals.
        Given in the same units (samples or time) as nn_x
    nn_interp_y: numpy array
        Location along the y-axis (i.e., actual values) of the interpolated NN signal.
        Given in the same units as nn_y
        obtained.
        
    References
    ----------
    A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """
    
    cs = CubicSpline(nn_x, nn_y)
    
    nn_interp_x = np.arange(from_time, to_time, T_int)
    nn_interp_y = cs(nn_interp_x)

    return nn_interp_x, nn_interp_y