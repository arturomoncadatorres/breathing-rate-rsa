# -*- coding: utf-8 -*-
"""
helpers.py
Helper functions.

@author: Arturo Moncada-Torres
"""

import numpy as np

#%%
def find_closest_idx(x, value):
    """ 
    Find the index of an array that corresponds to the closest given value.
    
    Parameters
    ----------
    x: list or np.array
        Iterable to search
    value: float
        Value of interest
        
    Returns
    -------
    idx: int
        Index that corresponds to the closest given value.
    """    
    # Find the closest index corresponding to the time of the original signal.
    diff_array = np.absolute(x - value)
    idx = diff_array.argmin()
    
    return idx