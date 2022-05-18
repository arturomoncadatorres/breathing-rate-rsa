# -*- coding: utf-8 -*-
"""
rr.py
Functions for manipulation of the RR signal (i.e., time between R peaks)

@author: Arturo Moncada-Torres
"""
import numpy as np

#%%
def calculate_nn(r_peaks):
    """ 
    Calculate the NN intervals.
    
    Parameters
    ----------
    r_peaks: numpy array
        Location (either in samples or time) of the R peaks of an ECG signal.
        
    Returns
    -------
    nn_x: numpy array
        Location along the x-axis of the NN intervals.
        Given in the same units (samples or time) as r_peaks
    nn_y: numpy array
        Location along the y-axis (i.e., actual values) of the NN intervals.
        Given in the same units as the ECG signal from which r_peaks was
        obtained.
    """

    # The NN intervals are located between two R peaks. 
    # We can compute them by averaging the time (or samples) between them.
    # To do this easily, we create a copy of the R peaks array and shift them
    # by one. This way, we can compute the average between both arrays.

    r_peaks_shifted = r_peaks[1:]
    r_peaks_shifted = np.append(r_peaks_shifted, np.nan) # Note that we need to append a NaN at the end to match the length.
    
    nn_x = (r_peaks + r_peaks_shifted) / 2
    
    # The actual NN values are just the difference between consecutive R peaks.
    nn_y = r_peaks_shifted - r_peaks
    
    # Remove the last element (which was a NaN)
    nn_x = nn_x[:-1]
    nn_y = nn_y[:-1]
    
    return nn_x, nn_y