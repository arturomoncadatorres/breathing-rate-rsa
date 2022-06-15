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
import itertools

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
        ax.set_ylabel("PSD [dB]")
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


#%%
def acf_adv(nn, fs, f_low=0.1, f_high=0.5, visualizations=False):
    """ 
    Compute the breathing rate using the Autocorrelation Advanced method
    (ACF-adv in the original paper).
    
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
    breathing_rate: float [Hz]
        Computed breathing rate. 
    fig, ax: matplotlib figure and axes
        If visualizations is set to False, these will be np.nan
    
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """
    
    # Calculate the differences of subsequent intervals
    nn_shifted = nn[1:]
    nn_shifted = np.append(nn_shifted, np.nan) # Note that we need to append a NaN at the end to match the length.
    nn_delta = nn_shifted - nn
    nn_delta = nn_delta[:-1]
    
    # Compute the ACF for all lags |delta| <= n/2
    acf = sm.tsa.acf(nn_delta, nlags=len(nn)//2)
    
    
    # Important parameters
    N = len(nn) # Number of points
    Ts = 1/fs # Sampling period
    
    # Compute the FFT
    fft = sp.fft.fft(acf, N)
    
    # Compute the magnitude spectrum (and limit it to the positive side)
    fft_mag = np.abs(fft[:N//2])
    
    # Remove DC component (f = 0).
    # Notice that this isn't explicitely mentioned in the paper, but
    # I believe it makes senes to remove it, since we didn't perform
    # any pre-processing to remove it.
    fft_mag[0] = 0
    
    # Compute the PSD (with proper scaling).
    psd = fft_mag ** 2    
    psd_scaled = psd / (fs * N)
    psd_scaled[1:-1] = 2 * psd_scaled[1:-1]
    
    
    # Compute the frequency axis (and limit it to the positive side)
    f = sp.fft.fftfreq(N, Ts)[:N//2]
    
    # Find band of interest.
    f_low_idx = helpers.find_closest_idx(f, f_low)
    f_high_idx = helpers.find_closest_idx(f, f_high)
    f_band = f[f_low_idx:f_high_idx]
    
    # Compute the median of the PSD in the band of interest
    psd_band = psd[f_low_idx:f_high_idx]
    psd_band_median = np.median(psd_band)
    
    # Compute the weighted average of the frequency band of relevance.
    f_relevant = f_band[psd_band > psd_band_median]
    psd_band_weighted = psd_band[psd_band > psd_band_median] / np.sum(psd_band[psd_band > psd_band_median])
    breathing_rate = np.sum(f_relevant * psd_band_weighted)

    if visualizations:
        fig, ax = plt.subplots(1, 1, figsize=[7,5])
        plt.plot(f, 10*np.log10(psd))
        plt.plot(f_band, 10*np.log10(psd_band))
        plt.axvline(x=f_low, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axvline(x=f_high, color=[0.6, 0.6, 0.6], linestyle='--')
        plt.axvline(x=breathing_rate, color=np.array([20, 102, 84])/255., linestyle='--')
        plt.axhline(y=10*np.log10(psd_band_median), color=[0.6, 0.6, 0.6], linestyle=':')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dB]")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)     
    else:
        fig = np.nan
        ax = np.nan
        
    return breathing_rate, fig, ax


#%%
def count_orig(sig, fs, f_low=0.1, f_high=0.5, visualizations=False):
    """ 
    Compute the breathing rate using the Original Counting method
    (Count-orig in the original paper).
    
    Parameters
    ----------
    sig:
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
    
    #%%

    # 1. Filter the signal of interest by a BPF.
    b, a = sp.signal.butter(6, [f_low, f_high], btype='bandpass', analog=False, fs=fs)
    w, h = sp.signal.freqz(b, a, fs=fs)
    
    sig_filtered = sp.signal.filtfilt(b, a, sig)
      
    
    # 2. Find local maxima and minima.
    maxima_loc, _ = sp.signal.find_peaks(sig_filtered)
    maxima = sig_filtered[maxima_loc]
    
    minima_loc, _ = sp.signal.find_peaks(-sig_filtered)
    minima = sig_filtered[minima_loc]
    

    # 3. Find threshold
    q3 = np.percentile(maxima, 75)
    threshold = q3 * 0.2


    # 4. Find valid breathing cycles
    maxima_valid = maxima[maxima > threshold]
    maxima_valid_loc = maxima_loc[maxima > threshold]
    
    segments_valid = []
    for ii, maximum_valid_loc in enumerate(maxima_valid_loc[:-1]):

        maximum_valid_loc2 = maxima_valid_loc[ii+1]
        
        # Check for exactly one minimum...
        minima_bool = (minima_loc > maximum_valid_loc) & (minima_loc < maximum_valid_loc2)
        if sum(minima_bool) != 1:
            continue
        
        # ...below zero
        minimum = minima[(minima_loc > maximum_valid_loc) & (minima_loc < maximum_valid_loc2)][0]
        if minimum >= 0:
            continue
        
        # Check for no other local extrema (i.e., other local maxima)
        maxima_bool = (maxima_loc > maximum_valid_loc) & (maxima_loc < maximum_valid_loc2)
        if sum(maxima_bool) > 0:
            continue
        
        # If the segment passed all conditions, we know it is valid.
        idx = np.arange(maximum_valid_loc, maximum_valid_loc2, 1)
        segment = {'idx':idx, 'values':sig_filtered[idx], 'n':len(idx)}
        
        segments_valid.append(segment)

        
    # 5. Compute the breathing rate as the reciprocal of the average of 
    # all detected respiratory cycles.
    mean_n = sum(x['n'] for x in segments_valid) / len(segments_valid)
    mean_n_s = mean_n / fs # [samples] --> [s]
    breathing_rate = 1/mean_n_s        
        
    if visualizations:
        fig, ax = plt.subplots(2, 1, figsize=[7,5], sharex=True)
        ax[0].plot(np.arange(0, len(sig_filtered))/fs, sig_filtered, color='C0', linewidth=2)
        ax[0].plot(maxima_loc/fs, maxima, marker='o', color='0.75', markersize=4, linestyle='None')
        ax[0].plot(maxima_valid_loc/fs, maxima_valid, marker='o', color='red', markersize=4, linestyle='None')
        ax[0].plot(minima_loc/fs, minima, marker='*', color='black', markersize=4, linestyle='None')
        ax[0].axhline(y=0, color=[0.6, 0.6, 0.6], linestyle=':', linewidth=1)
        ax[0].axhline(y=threshold, color=[0.6, 0.6, 0.6], linestyle='--', linewidth=1)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)    

        for segment_valid in segments_valid:
            ax[1].plot(segment_valid['idx']/fs, segment_valid['values'], color='C0', linewidth=2)      
        ax[1].axhline(y=0, color=[0.6, 0.6, 0.6], linestyle=':', linewidth=1)
        ax[1].axhline(y=threshold, color=[0.6, 0.6, 0.6], linestyle='--', linewidth=1)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)    
        ax[1].set_xlabel('Time [s]')
        fig.supylabel('Amplitude')

        plt.show()
    else:
        fig = np.nan
        ax = np.nan
        
    return breathing_rate, fig, ax
    

#%%
def count_adv(sig, fs, f_low=0.1, f_high=0.5, signal_type='nn', visualizations=False):
    """ 
    Compute the breathing rate using the Advanced Counting method
    (Count-adv in the original paper).
    
    Parameters
    ----------
    sig:
    fs: float [Hz]
        Sampling rate
    f_low, f_high: float [Hz]
        Bottom/top boundaries of the band of interest.
        Defaults to 0.1 and 0.5 Hz, as suggested in the original paper.
    signal_type: string
        Type of signal given as an input. Possible values are
            nn              NN interval (default)
            resp            Respiration signal
            respiration     Same as resp
            
        This is needed to choose the appropriate threshold (step 3)
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
    
    #%%

    # 1. Filter the signal of interest by a BPF.
    b, a = sp.signal.butter(6, [f_low, f_high], btype='bandpass', analog=False, fs=fs)
    w, h = sp.signal.freqz(b, a, fs=fs)
    
    sig_filtered = sp.signal.filtfilt(b, a, sig)

    
    # 2. Find local maxima and minima.
    maxima_loc, _ = sp.signal.find_peaks(sig_filtered)
    maxima = sig_filtered[maxima_loc]
    
    minima_loc, _ = sp.signal.find_peaks(-sig_filtered)
    minima = sig_filtered[minima_loc]

        
    # 3. Calculate vertical differences and find threshold
    # We assume that a maxima and minima alternate
    
    # First, we interweave the arrays. This allows us to calculate the
    # differences very easily.
    max_min = np.hstack(itertools.zip_longest(maxima, minima, fillvalue=np.nan)) 
    
    # In case the array lenghts are different, we will have a np.nan.
    # We will remove it.
    max_min = max_min[~np.isnan(max_min)]
    
    # We create a shifted copy of the difference array and perform the
    # subtraction.
    max_min_shifted = max_min[1:]
    max_min_differences = max_min[:-1] - max_min_shifted
    max_min_abs_differences = np.abs(max_min_differences)
    
    # Compute the threshold, which depends on the signal type.
    q3 = np.quantile(max_min_abs_differences, 0.75)
    if signal_type == 'nn':
        threshold = q3 * 0.1
    elif (signal_type == 'resp') or (signal_type == 'respiration'):
        threshold = q3 * 0.3
        
    
    # 4. Remove extrema pair until all differences are larger than the 
    # threshold.
    while sum(max_min_abs_differences < threshold):
        
        for ii, _ in enumerate(max_min[:-1]):
            
            # Compute the difference between min and max.
            # If it is smaller than the threshold, kick out that segment
            # by filling it with np.nan...
            diff = max_min[ii] - max_min[ii+1]
            if np.abs(diff) < threshold:
                max_min[ii] = np.nan
                max_min[ii+1] = np.nan

        # ...and removing it at the end of the iteration.
        max_min = max_min[~np.isnan(max_min)]
               
        # Compute the differences again to see if there are still differences
        # smaller than the threshold. When there are none, we get out
        # of the loop.
        max_min_shifted = max_min[1:]
        max_min_differences = max_min[:-1] - max_min_shifted
        max_min_abs_differences = np.abs(max_min_differences)
        
        
    # 5. Compute the breathing rate as the reciprocal of the average of 
    # all detected respiratory cycles.
    
    # Untangle the max and min
    maxima_valid_bool = np.isin(maxima, max_min)
    maxima_valid_idx = maxima_loc[maxima_valid_bool]
    maxima_valid = maxima[maxima_valid_bool]
    
    minima_valid_bool = np.isin(minima, max_min)
    minima_valid_idx = minima_loc[minima_valid_bool]
    minima_valid = minima[minima_valid_bool]
        
    # For consistency, we will use the same technique as in count_orig
    segments_valid = []
    for ii, maximum_valid_idx in enumerate(maxima_valid_idx[:-1]):
    
        maximum_valid_idx2 = maxima_valid_idx[ii+1]
        
        idx = np.arange(maximum_valid_idx, maximum_valid_idx2, 1)
        segment = {'idx':idx, 'values':sig_filtered[idx], 'n':len(idx)}
        
        segments_valid.append(segment)

    mean_n = sum(x['n'] for x in segments_valid) / len(segments_valid)
    mean_n_s = mean_n / fs # [samples] --> [s]
    breathing_rate = 1/mean_n_s        
        
    if visualizations:
        fig, ax = plt.subplots(2, 1, figsize=[7,5], sharex=True)
        ax[0].plot(np.arange(0, len(sig_filtered))/fs, sig_filtered, color='C0', linewidth=2)
        ax[0].plot(maxima_loc/fs, maxima, marker='o', color='0.75', markersize=4, linestyle='None')
        ax[0].plot(maxima_valid_idx/fs, maxima_valid, marker='o', color='red', markersize=4, linestyle='None')
        ax[0].plot(minima_loc/fs, minima, marker='*', color='0.75', markersize=4, linestyle='None')
        ax[0].plot(minima_valid_idx/fs, minima_valid, marker='*', color='black', markersize=4, linestyle='None')
        ax[0].axhline(y=0, color=[0.6, 0.6, 0.6], linestyle=':', linewidth=1)
        ax[0].axhline(y=threshold, color=[0.6, 0.6, 0.6], linestyle='--', linewidth=1)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)    
    
        for segment_valid in segments_valid:
            ax[1].plot(segment_valid['idx']/fs, segment_valid['values'], color='C0', linewidth=2)      
        ax[1].axhline(y=0, color=[0.6, 0.6, 0.6], linestyle=':', linewidth=1)
        ax[1].axhline(y=threshold, color=[0.6, 0.6, 0.6], linestyle='--', linewidth=1)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)    
        ax[1].set_xlabel('Time [s]')
        fig.supylabel('Amplitude')
    
        plt.show()
    else:
        fig = np.nan
        ax = np.nan
        
    return breathing_rate, fig, ax