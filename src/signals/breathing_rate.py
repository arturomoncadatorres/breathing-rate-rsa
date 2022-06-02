# -*- coding: utf-8 -*-
"""
breathing_rate.py
Functions for calculating the breathing rate as described in
A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
Sinus Arrhythmia: Comparison of Various Methods", 2008.

@author: Arturo Moncada-Torres
"""

import numpy as np
import scipy as sp
import spectrum
from statsmodels import api as sm
# from statsmodels.graphics.tsaplots import plot_acf

import matplotlib as mpl
from matplotlib import pyplot as plt

import src.utils.helpers as helpers

#%%
def spec_ft(nn, fs, f_low=0.1, f_high=0.5, visualizations=False):
    """ 
    Compute the breathing rate using the Fourier Analysis method
    (Spec-FT in the original paper).
    
    Parameters
    ----------
    nn:
    fs: float [Hz]
        Sampling rate
    f_low, f_high: float [Hz]
        Bottom/top boundaries of the band of interest.
        Defaults to 0.1 and 0.5 Hz, as suggested in the original paper.
    visualizations: Boolean
        Define if plots will be generated (True) or not (False, default)
    
        
    Returns
    -------
    breathing_rate: float
        Computed breathing rate. If conditions are not met,
        this could be a np.nan
    fig, ax: matplotlib figure and axes
        If visualizations is set to False, these will be np.nan
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    - Heckmann, C., and D. Schenk. Klinisch-rhythmologische Aspekte der 
    Momentanherzfrequenz-Analyse aus dem 24-h-EKG. Herzmedizin 8:134–145, 1985
    """    
    
    # Important parameters
    N = len(nn) # Number of points
    Ts = 1/fs # Sampling period
    
    # Compute the FFT
    fft = sp.fft.fft(nn, N)
    
    # Compute the magnitude spectrum (and limit it to the positive side)
    fft_mag = np.abs(fft[:N//2])
    
    # Remove DC component (f = 0).
    # Notice that this isn't explicitely mentioned in the paper, but
    # I believe it makes senes to remove it, since we didn't perform
    # any pre-processing to remove it.
    fft_mag[0] = 0
    
    # Compute the frequency axis (and limit it to the positive side)
    f = sp.fft.fftfreq(N, Ts)[:N//2]
    
    # Find band of interest.
    f_low_idx = helpers.find_closest_idx(f, f_low)
    f_high_idx = helpers.find_closest_idx(f, f_high)

    # Compute the mean breathing rate as the frequency of the
    # maximum of the (absolute of the) band of interest
    # (between 0.1 and 0.5 Hz), a.k.a. |Xm|
    xm_peak = np.max(fft_mag[f_low_idx:f_high_idx])
    breathing_rate_tmp = f[fft_mag == xm_peak][0]
    
    # Check if both conditions are fulfilled.
    # - |Xm| is larger than the mean + 1 SD of the entire spectrum
    # - The difference between |Xm| and the second largest component
    # should be >=10% of |Xm|
    # or
    # The frequency of |Xm| is separated from the frequency of the
    # absolute maximum by <0.01 Hz.
    threshold = np.mean(fft_mag) + np.std(fft_mag)
    xm_peak2 = np.partition(fft_mag.flatten(), -2)[-2]
    abs_max_f = f[np.argmax(fft_mag)]
    
    if (xm_peak > threshold) and (((xm_peak-xm_peak2) > xm_peak*0.01) or (np.abs(abs_max_f - breathing_rate_tmp)<0.01)):
        breathing_rate = breathing_rate_tmp
        
    # Otherwise, computed breathing rate is not valid.
    else:
        breathing_rate = np.nan
    
    # If needed, generate visualizations.
    if visualizations == True:
        fig, ax = plt.subplots(1, 1, figsize=[7,5])
        plt.plot(f, fft_mag)
        plt.plot(breathing_rate_tmp, xm_peak, marker='o', color=[1,0,0])
        plt.axvline(x=f_low, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axvline(x=f_high, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axhline(y=threshold, color=[0.6, 0.6, 0.6], linestyle=':')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("$| Xm |$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
    else:
        fig = np.nan
        ax = np.nan
    
    return breathing_rate, fig, ax


#%%
def spec_ar(nn, fs, f_low=0.1, f_high=0.5, visualizations=False):
    """ 
    Compute the breathing rate using the Autoregressive Modeling method
    (Spec-AR in the original paper).
    
    Parameters
    ----------
    nn:
    fs: float [Hz]
        Sampling rate
    f_low, f_high: float [Hz]
        Bottom/top boundaries of the band of interest.
        Defaults to 0.1 and 0.5 Hz, as suggested in the original paper.
    visualizations: Boolean
        Define if plots will be generated (True) or not (False, default)
    
        
    Returns
    -------
    breathing_rate: float
        Computed breathing rate. If conditions are not met,
        this could be a np.nan
    fig, ax: matplotlib figure and axes
        If visualizations is set to False, these will be np.nan
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    - HFru¨ hwirth, M. Methods for calculation of respiration rate
    and respiratory sinus arrhythmia from heart rate variability
    (in German). Master’s thesis, Technical University of
    Graz, Austria, 2003. (I wasn't able to read this find nor read this reference)
    - Cokelaer et al, (2017), 'Spectrum': Spectral Analysis in Python, 
    Journal of Open Source Software, 2(18), 348, doi:10.21105/joss.00348
    - https://pyspectrum.readthedocs.io/en/latest/tutorial_pburg.html
    - https://dsp.stackexchange.com/questions/32187/what-should-be-the-correct-scaling-for-psd-calculation-using-tt-fft
    - https://dsp.stackexchange.com/questions/27213/power-spectrum-estimate-from-fft/27218#27218
    """
    
    # Reduce interpolation sample rate by half by dropping every second sample.
    # If done according to the paper, this means it goes from 5 to 2.5 Hz.
    nn_undersampled = nn[::2]
    N = len(nn_undersampled)
    
    fs_undersampled = fs/2
    

    # Remove mean trend, as recommended by Spectrum (https://pyspectrum.readthedocs.io/en/latest/ref_param.html#spectrum.burg.arburg)
    # (and aligns with intuition, since DC component is not useful in this case).
    nn_detrended = nn_undersampled - np.mean(nn_undersampled)
    
    # Generate an autoregressive model to obtain the powed spectrum 
    # density (PSD) using a Burg algorithm with a model of order 15.
    # This is easily done using Spectrum.
    order = 15
    ar, rho, refl_coeff = spectrum.arburg(nn_detrended, order)
    # ar is an array of the complex autoregressive parameters
    # rho represents the driving noise variance (mean square of residual
    # noise) from the whitening operation of the Burg filter
    # refl_coeff contains the reflection coefficients defining the filter of the model.
    
    psd_ = spectrum.arma2psd(ar, rho=rho, T=fs_undersampled, NFFT=N)
    
    psd = psd_[len(psd_)//2:len(psd_)]
    psd = psd[::-1]
    # Notice that psd_ is sliced in a peculiar way. That is because
    # the way Spectrum gives the PSD as an output (two-sided)
    # and we are only interested in the positive part.
    
    # Perform scaling of the PSD
    # Notice we do *not* scale the DC and the Nyquist components.
    # https://dsp.stackexchange.com/questions/32187/what-should-be-the-correct-scaling-for-psd-calculation-using-tt-fft
    # https://dsp.stackexchange.com/questions/27213/power-spectrum-estimate-from-fft/27218#27218
    psd_scaled = psd / (fs_undersampled * N)
    psd_scaled[1:-1] = 2 * psd_scaled[1:-1]
    
    # Find band of interest.
    f = np.arange(0, (fs_undersampled/2), fs_undersampled/N)
    f_low_idx = helpers.find_closest_idx(f, f_low)
    f_high_idx = helpers.find_closest_idx(f, f_high)
    
    # Compute the mean breathing rate as the frequency component
    # in the band of interest (between 0.1 and 0.5 Hz) which exhibits
    # a spectral power >=5.5% of the total power.
    xm2_peak = np.max(psd_scaled[f_low_idx:f_high_idx])
    breathing_rate_tmp = f[psd_scaled == xm2_peak][0]
    
    threshold = np.sum(psd_scaled) * 0.055
    if xm2_peak > threshold:
        breathing_rate = breathing_rate_tmp
    else:
        breathing_rate = np.nan

    # If needed, generate visualizations.
    if visualizations == True:
        
        fig, ax = plt.subplots(1, 1, figsize=[7,5])
        plt.plot(f, 10*np.log10(psd_scaled))
        plt.plot(breathing_rate_tmp, 10*np.log10(xm2_peak), marker='o', color=[1,0,0])
        plt.axvline(x=f_low, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axvline(x=f_high, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axhline(y=10*np.log10(threshold), color=[0.6, 0.6, 0.6], linestyle=':')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("$| Xm |^2$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
    else:
        fig = np.nan
        ax = np.nan
        
    return breathing_rate, fig, ax

    
#%%
def acf_max(nn, fs, f_low=0.1, f_high=0.5, visualizations=False):
    """ 
    Compute the breathing rate using the Autocorrelation Maximum method
    (ACF-max in the original paper).
    
    Parameters
    ----------
    nn:
    fs: float [Hz]
        Sampling rate
    f_low, f_high: float [Hz]
        Bottom/top boundaries of the band of interest.
        Defaults to 0.1 and 0.5 Hz, as suggested in the original paper.
    visualizations: Boolean
        Define if plots will be generated (True) or not (False, default)
    
        
    Returns
    -------
    breathing_rate: float
        Computed breathing rate. If conditions are not met,
        this could be a np.nan
    fig, ax: matplotlib figure and axes
        If visualizations is set to False, these will be np.nan
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """
    
    acf = sm.tsa.acf(nn, nlags=len(nn))
    lags = np.arange(len(acf))
    
    # Perform interpolation to improve resolution.
    # The paper doesn't give particular specifications of the interpolation,
    # so we will just make the resolution x10 better.
    cs = sp.interpolate.CubicSpline(lags, acf)
    lags_interp = np.arange(0, len(lags), 1/10)
    acf_interp = cs(lags_interp)
    
    
    # Find the peaks of the ACF.
    peaks, _ = sp.signal.find_peaks(acf_interp)
    
    # Find the first peak of the ACF.
    peak_loc = peaks[0]
    peak = acf_interp[peak_loc]
    
    # Compute breathing rate.
    # Notice we divide by an additional factor of 10 due to to the
    # interpolation.
    breathing_rate = peak_loc * (1/(fs*10))
    
    if visualizations:
        fig, ax = plt.subplots(1, 1, figsize=[7,5])
        plt.plot(lags_interp, acf_interp)
        plt.plot(lags_interp[peaks[1:]], acf_interp[peaks[1:]], 'ro')
        plt.plot(lags_interp[peak_loc], peak, 'r*', markersize=10)
        ax.set_xlabel("Lag [samples]")
        ax.set_ylabel("ACF")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)     
        plt.show()
    
    else:
        fig = np.nan
        ax = np.nan
        
    return breathing_rate, fig, ax


