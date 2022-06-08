#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
from collections import OrderedDict

from hypyp import viz
import matplotlib

import mne
import numpy as np
# Import custom tools
from utils import basicAnalysis_tools

mne.set_log_level('warning')
# matplotlib.use('Qt5Agg')




# %%
# viz.viz_2D_topomap_inter(epo1, epo2, C, threshold='auto', steps=10, lab=True)

#%%
# viz.viz_3D_inter(epo1, epo2, C, threshold='auto', steps=10, lab=False)