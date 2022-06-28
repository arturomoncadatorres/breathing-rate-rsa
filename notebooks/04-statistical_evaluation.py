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
# # `03-statistical_evaluation`
# In this notebook, we perform the statistical evaluation to compare
# the different estimators.
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

