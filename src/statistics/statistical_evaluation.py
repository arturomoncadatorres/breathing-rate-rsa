# -*- coding: utf-8 -*-
"""
statistical_evaluation.py
Functions for performing statistical evaluation as described in
A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
Sinus Arrhythmia: Comparison of Various Methods", 2008.

@author: Arturo Moncada-Torres
"""

import numpy as np

#%%
def compute_rp(x, y):
    """ 
    Compute the Pearson correlation between x and y.
    
    Parameters
    ----------
    x: numpy array
    y: numpy array
    
        
    Returns
    -------
    rp: float
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    
    rp = np.mean(((x-mu_x)*(y-mu_y))) / (std_x * std_y)
    # Notice this yields the same results as
    # np.corrcoef(x.T, y.T)[0,1]
    
    return rp
    

#%%
def compute_rc(x, y):
    """ 
    Compute the corcordance correlation coefficient between x and y.
    
    Parameters
    ----------
    x: numpy array
    y: numpy array
    
        
    Returns
    -------
    rc: float
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    - Lin, L. I.-K. A concordance correlation coefficient to evaluate 
    reproducibility. Biometrics 45:255–268, 1989
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    
    rc = 2*np.mean(((x-mu_x)*(y-mu_y))) / (std_x**2 + std_y**2 + (mu_y - mu_x)**2)
    
    return rc
    

#%%
def compute_v(x, y):
    """ 
    Compute the scale (i.e., slope of the linear correlation between x and y).
    Optimum value is 1.
    
    Parameters
    ----------
    x: numpy array
    y: numpy array
    
        
    Returns
    -------
    rc: float
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    - Lin, L. I.-K. A concordance correlation coefficient to evaluate 
    reproducibility. Biometrics 45:255–268, 1989
    """   
    std_x = np.std(x)
    std_y = np.std(y)
    
    
    v = std_y / std_x
    
    return v


#%%
def compute_u(x, y):
    """ 
    Compute the location (i.e., intercept of the linear correlation between x and y).
    Optimum value is 0.
    
    Parameters
    ----------
    x: numpy array
    y: numpy array
    
        
    Returns
    -------
    rc: float
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """   
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    
    u = (mu_y - mu_x) / np.sqrt(std_x * std_y)
    
    return u