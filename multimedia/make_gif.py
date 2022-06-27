# -*- coding: utf-8 -*-
"""
make_gif.py
Script for making the GIF used in the project's README.

@author: Arturo Moncada-Torres
"""
import io
import pathlib
import pickle
import scipy as sp

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 600
from PIL import Image

#%%
# Read GIF
image = Image.open("./lungs.gif")
n_frames = image.n_frames


#%%
# Read signals
PATH_DATA = pathlib.Path('../data/processed')

# We choose this dataset because it looks very clean, perfect for a demo.
participant_id = 'f2y10'
with open(str(PATH_DATA/(participant_id + '.pkl')), 'rb') as f:
    dataset = pickle.load(f)

# Trim the respiration signal.
# Additionally, for easier peak finding, we will also smooth it with
# a mean filter with window size of 100 (size defined empirically).
respiration = dataset['respiration_3'][30].copy()
respiration = respiration[0:2000]
respiration_smooth = sp.ndimage.filters.uniform_filter1d(respiration, 100)

# Trim the ECG signal.
ecg = dataset['ecg_3'][30].copy()
ecg = ecg[0:2000]

# We will make the GIF of only one breathing cycle (the first, to be precise).
minima, _ = sp.signal.find_peaks(-respiration_smooth, distance=100)
respiration_short = respiration[minima[1]:minima[2]]
respiration_smooth_short = respiration_smooth[minima[1]:minima[2]]
ecg_short = ecg[minima[1]:minima[2]]

# For demonstration purposes, we will resample the respiration signal
# so that each frame corresponds to ~100 samples.
factor = 100
respiration_resampled = sp.signal.resample(respiration_short, n_frames*factor)
respiration_smooth_resampled = sp.signal.resample(respiration_smooth_short, n_frames*factor)
ecg_resampled = sp.signal.resample(ecg_short, n_frames*factor)


#%%
# In order to generate the GIF, we need to save a figure in memory
# as a PIL image. For that, we can create a handy function.
# See original implementation in
# https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

#%%
# Create visualization.

images_gif = []

for ii in range(n_frames):

    # Temporal plotting style.    
    with plt.rc_context({'axes.edgecolor':'0.75', 'xtick.color':'0.75'}):
        fig, ax = plt.subplot_mosaic('AB;AC', figsize=[10,5])
        
        print(f"Processing frame {ii}...")
        
        # Lung GIF frame.
        image.seek(ii)
        frame_current = image.convert('RGBA')
        ax['A'].imshow(frame_current)
        ax['A'].set_axis_off()
        
        # Respiration signal (with moving marker)
        ax['B'].plot(respiration_resampled, linewidth=3, color='0.8')
        ax['B'].plot(respiration_smooth_resampled, linewidth=2, color='#00B3A2')
        ax['B'].axvline(ii*factor, color='0.6', linestyle='--', linewidth=1)
        ax['B'].plot(ii*factor, respiration_smooth_resampled[ii*factor], marker=7, markersize=10, color='#11557C')
        ax['B'].set_xlim([0, len(respiration_smooth_resampled)-1])
        ax['B'].spines['top'].set_visible(False)
        ax['B'].spines['right'].set_visible(False)
        ax['B'].tick_params(labelbottom=False)
        ax['B'].tick_params(labelleft=False)
        ax['B'].tick_params(left=False)
        ax['B'].set_ylabel("Respiration")
        
        # ECG signal
        ax['C'].plot(ecg_resampled, linewidth=2, color='#00B3A2')
        ax['C'].axvline(ii*factor, color='0.6', linestyle='--', linewidth=1)
        ax['C'].set_xlim([0, len(ecg_resampled)-1])
        ax['C'].spines['top'].set_visible(False)
        ax['C'].spines['right'].set_visible(False)
        ax['C'].tick_params(labelbottom=False)
        ax['C'].tick_params(labelleft=False)
        ax['C'].tick_params(left=False)
        ax['C'].set_xlabel("Time")
        ax['C'].set_ylabel("ECG")
        
        # Display and save in memory for GIF creation.
        plt.show()
        images_gif.append(fig2img(fig))
        
# Create GIF.
# See pillow's documentation for saving GIFs
# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
images_gif[0].save('./respiration_animation.gif', save_all=True, append_images=images_gif[1:], optimize=False, duration=4000/n_frames, loop=0)
