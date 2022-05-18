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
# # `02-preprocessing`
# In this notebook, we showcase the pre-processing steps
#
# ## Preliminaries
# Imports
# %%
import pathlib
import wfdb
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 600

import src.signals.rr as rr
import src.signals.nn as nn
import src.signals.respiration as resp


# %% [markdown]
# Define paths (based on the Cookiecutter file structure)
# %%
PATH_DATA = pathlib.Path('../data')

# %% [markdown]
# ## Peek into the data
# We will take a quick look at how some of the signals look like.
# For now, we will only take a look at 10,000 samples.

# %%
sample_from = 0
sample_to = 10_000

# %% [markdown]
# We can read a record with [`wfdb.rdrecord`](https://wfdb.readthedocs.io/en/latest/wfdb.html#wfdb.rdrecord),
# annotations with [`wfdb.rdann`](https://wfdb.readthedocs.io/en/latest/wfdb.html#wfdb.rdann)
# and easily plot them with [`wfdb.plot_wfdb`](https://wfdb.readthedocs.io/en/latest/wfdb.html#wfdb.plot_wfdb).
# Notice how we some matplotlib for tweaking the plot.
#
# Notice how in this case, the annotations correspond to the R peaks marked
# on top of the respiration signal.

# %%
participant = 'f1y01'
record = wfdb.rdrecord(str(PATH_DATA/'raw'/participant), sampfrom=sample_from, sampto=sample_to)
annotation = wfdb.rdann(str(PATH_DATA/'raw'/participant), 'ecg', sampfrom=sample_from, sampto=sample_to)

fig = wfdb.plot_wfdb(record=record, title='Sample signals', annotation=annotation, return_fig=True) 
ax = fig.axes
ax[1].set_xlabel("Samples")

ax[0].set_ylabel("Respiration [mV]")
ax[1].set_ylabel("ECG [mV]")

print(record.__dict__)

# %% [markdown]
# A couple of important variables for further use.

# %%
fs = record.fs # Sampling rate
Ts = 1/fs # Sampling period
n_samples = record.sig_len # Total number of samples
t = np.arange(0, n_samples*Ts, Ts)

# %% [markdown]
# In this case, the annotations correspond to the R peaks of the ECG marked
# on top of the respiration signal.
#
# We can also take a peek into the annotations. 
# A complete description of what the annotations mean can be found [here](https://github.com/MIT-LCP/wfdb-python/blob/0d42dfb4b2946625f00cbf500d830d374a201153/wfdb/io/annotation.py#L14)

# %%
annotations = annotation.symbol
r_peaks_samples = annotation.sample

print(annotations[0:10])
print(r_peaks_samples[0:10])
print(set(annotations))


# %% [markdown]
# ## Pre-processing
# ### Compute NN intervals

# %%
r_peaks_t = r_peaks_samples * Ts # [samples] --> [s]
nn_x, nn_y = rr.calculate_nn(r_peaks_t)

# %% [markdown]
# ### Interpolate NN intervals

# %%
T_int = 0.2 # Interpolation interval [s]
last_time_sample = n_samples * Ts
nn_interp_x, nn_interp_y = nn.interpolate(nn_x, nn_y, from_time=0, to_time=last_time_sample, T_int=T_int)

# %% [markdown]
# ### Smooth respiration signal

# %%
respiration = record.p_signal[:,0]
respiration_smooth_x, respiration_smooth_y = resp.smooth(respiration, nn_interp_x, from_time=0, to_time=last_time_sample, Ts=Ts, T_int=T_int)

# %% [markdown]
# ## Visualizations
# These are similar to Fig. 1b, 1c, and 1d of the original paper.

# %%
fig, ax = plt.subplots(2, 1, figsize=[10,5], sharex=True)
ax[0].plot(nn_interp_x, nn_interp_y)
ax[0].plot(nn_x, nn_y, marker='o', linestyle='None')
ax[0].set_ylim([0.7, 0.9])
ax[0].set_ylabel("NN series [ms]")

ax[1].plot(t, respiration)
ax[1].plot(respiration_smooth_x, respiration_smooth_y+1)
ax[1].set_xlim([0, 40])
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Respiration [mV]")

# %% [markdown]
# The top panel shows the NN series. The dots correspond to the original
# points, while the line corresponds to the interpolated signal.
#
# The bottom panel shows the respiration signals. In blue, the original
# (noisy) signal, in orange the smoothened signal (shifted upwards artificially
# for clarity). 
#
# Notice how "the minima of the NN curve generally occur near the inspirational 
# maxima of breathing activity and *vice versa*".
#
# The rest of the pre-processing consists in splitting the original
# signals into parts of 5 min length, which isn't particularly interesting
# to showcase in this notebook.
