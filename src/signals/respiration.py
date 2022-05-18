# -*- coding: utf-8 -*-
"""
respiration.py
Functions for manipulation of the respiration signal.

@author: Arturo Moncada-Torres
"""

import numpy as np

#%%
def smooth(respiration, nn_interp_x, from_time, to_time, Ts, T_int=0.2):
    """ 
    Smooth the respiration signal using mean values around the
    points of the interpolated NN interval signal.
    
    Parameters
    ----------
    respiration: numpy array
        (Noisy) respiration signal
    nn_interp_x: numpy array
        Location (either in samples or time) of the interpolated NN intervals
    from_time: float
        Initial time (in seconds) of interest of the respiration signal
    to_time: float
        Final time (in seconds) of interest of the respiration signal
    Ts: float
        Sampling period
    T_int: float
        Interpolation interval. Defaults to 0.2 s, as recommended
        in the original paper (see References).
        
    Returns
    -------
    respiration_smooth_x: numpy array
        Location along the x-axis of the smoothened respiration signal.
    respiration_smooth_y: numpy array
        Smoothened respiration signal.
        
    References
    ----------
    A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """
    
    # Calculate the number of samples of the original respiration signal
    n_samples = len(respiration)
    
    # Initialize output.
    respiration_smooth_x = np.copy(nn_interp_x)
    respiration_smooth_y = np.zeros(len(nn_interp_x))
    

    # Create a time vector
    t = np.arange(from_time, to_time, Ts)
    
    # At each location of the interpolated NN signal...
    for ii, value in enumerate(nn_interp_x):
        
        # Find the closest index corresponding to the time of the original signal.
        diff_array = np.absolute(t - value)
        idx = diff_array.argmin()
        
        # Get the indexes around +/- T_int/2
        # Notice how they are bounded to avoid going below or above
        # the time limit.
        idx_low = max(from_time, round(idx - (T_int/2)*(1/Ts)))
        idx_high = min(n_samples, round(idx + (T_int/2)*(1/Ts)))
        
        # Compute the mean of the pertinent respiration signal samples.
        respiration_smooth_y[ii] = np.mean(respiration[idx_low:idx_high])
    
    return respiration_smooth_x, respiration_smooth_y