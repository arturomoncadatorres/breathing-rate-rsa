# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `03-analysis`
# In this notebook, we showcase the analysis techniques of the data.
# Specifically, we show the different methods used to calculate breathing rate.
#
# ## Preliminaries
# Imports

# %%
import scipy as sp
import pathlib
import pickle
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 600

import src.signals.breathing_rate as br


# %% [markdown]
# Define paths (based on the Cookiecutter file structure)

# %%
PATH_DATA = pathlib.Path('../data/processed')

# %% [markdown]
# ## Read data
# In this notebook, we will visualize one NN segment from one single participant
# for demo purposes. We chose this as an example since it is a relatively
# clean signal. However, feel free to experiment with data from different
# participants and different segments! This should give you an idea of
# the variability between the signals and how they affect the performance
# of the different methods.
#
# The script `calculate_breathing_rate` (`calculate_breathing_rate` in 
# the `Makefile`) processes the whole dataset.

# %%
participant_id = 'f2y10'
with open(str(PATH_DATA/(participant_id + '.pkl')), 'rb') as f:
    dataset = pickle.load(f)

nn = dataset['nn_interp_y_trimmed_3'][30]
fs = 1/dataset['T_int']  # Remember that the NN signal was interpolated with interval T_int.

# %% [markdown]
# ## Calculate breathing rate
# We will store the computeed breathing rates in a dictionary

# %%
breathing_rate = {}

# %% [markdown]
# ### Spectral analysis
# #### Fourier analysis (`Spec-FT`)

# %%
breathing_rate['spec_ft'], fix, ax = br.spec_ft(nn, fs, f_low=0.1, f_high=0.5, visualizations=True)
breathing_rate['spec_ft']

# %% [markdown]
# #### Autoregressive modelling (`Spec-AR`)

# %%
breathing_rate['spec_ar'], fig, ax = br.spec_ar(nn, fs, f_low=0.1, f_high=0.5, visualizations=True)
breathing_rate['spec_ar']

# %% [markdown]
# ### Autocorrelation function
# #### Autocorrelation maximum method (`ACF-max`)

# %%
breathing_rate['acf_max'], fig, ax = br.acf_max(nn, fs, f_low=0.1, f_high=0.5, visualizations=True)
breathing_rate['acf_max']

# %% [markdown]
# #### Autocorrelation advanced method (`ACF-adv`)

# %%
breathing_rate['acf_adv'], fig, ax = br.acf_adv(nn, fs, f_low=0.1, f_high=0.5, visualizations=True)
breathing_rate['acf_adv']

# %% [markdown]
# ### Counting methods
# Both of the counting methods process the signal using a Butterworth 
# BPF filter (step 1). For clarity, this is how the filter looks like:

# %%
f_low = 0.1 # [Hz]
f_high = 0.5 # [Hz]

b, a = sp.signal.butter(6, [f_low, f_high], btype='bandpass', analog=False, fs=fs)
w, h = sp.signal.freqz(b, a, fs=fs)

nn_filtered = sp.signal.filtfilt(b, a, nn)


# %% [markdown]
# For the sake of completion, this is how the filter looks like:

# %%
fig, ax = plt.subplots(1, 1, figsize=[7,5])
plt.semilogx(w, abs(h), label="Magnitude response")
plt.grid(which='both', axis='both')
plt.axvline(f_low, color='green', label="Frequency band of interest")
plt.axvline(f_high, color='green')
plt.title('Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)    
plt.show()


# %% [markdown]
# Notice that the original paper uses a 10th order filter.
# However, we use `filtfilt` to filter the signal (to avoid distortion),
# which would result in passing the signal through two 5th order filters.
# Since we are using a BPF (LPF + HPF), a 5th order would result asymmetrical.
# Thus, we opted to use a 12th order filter.
#
# This is how the original and the filtered signal look like.

# %%
fig, ax = plt.subplots(1, 1, figsize=[7,5])
plt.plot(nn, label="Raw signal")
plt.plot(nn_filtered, label="Filtered signal")
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)    
plt.show()


# %% [markdown]
# #### Original counting method (`Count-orig`)

# %%
breathing_rate['count_orig'], fig, ax = br.count_orig(nn, fs, f_low=0.1, f_high=0.5, visualizations=True)
breathing_rate['count_orig']

# %% [markdown]
# #### Advanced counting method (`Count-adv`)

# %%
breathing_rate['count_adv'], fig, ax = br.count_adv(nn, fs, f_low=0.1, f_high=0.5, signal_type='nn', visualizations=True)
breathing_rate['count_adv']
