#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
# Import packages, paths, and useful variables.
from copy import deepcopy

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar

import mne
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch

import numpy as np
import pandas as pd
import seaborn as sns

# Import custom tools
from utils import basicAnalysis_tools
from utils.useful_variable import *

sns.set(rc={'axes.facecolor':'#f6f6f6', 'figure.facecolor':'white'})

#%% 
# Load data

data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/EEG_data_cleaned/"
save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"

psds_result_allFreq  = "results_psds_allFreq/"

expConditionDir = ['SNS_ES_cleaned', 'SNS_NS_cleaned']
conditions      = ['ES', 'NS']

#%% 

df_manifest = basicAnalysis_tools.get_analysis_manifest(data_path, expConditionDir, save_to=None)


# %%
# Make list of the Speaker and Listener
speaker_list  = []
listener_list   = []

for condition in df_manifest.keys(): 
# for condition in ['ES']:
    for dyad in df_manifest[condition].keys():
    # for dyad in ['4']:
        for i in [0, 1]: # Going through both files of the dyad/condition
            _,_,_,sub1,sub2,_,condi,_,whoswho = df_manifest[condition][dyad][i].split('_')        
            role = sub1[-1] if whoswho[0] == '1' else sub2[-1] # Look for whos is the sub in that file (speaker vs listener)
            speaker_list.append((df_manifest[condition][dyad][i],condi)) if role == 'S' else listener_list.append((df_manifest[condition][dyad][i],condi))

#%%
#           ###########   1. PSDs    ###########

# ############
# Compute PSDs 
# ############

input('sure you wanna do that?')

all_psds = {
    'ES':{
        "SPEAKER":[], 
        "LISTENER":[]}, 
    'NS':{
        "SPEAKER":[], 
        "LISTENER":[]}}

# Create similar dict structure to store amount of epochs per subject
mean_epoch_count = deepcopy(all_psds)

for condi in ['ES', 'NS']:
  
    for role in [(speaker_list, "SPEAKER"), (listener_list, "LISTENER")]:
        
        print("\n-------", condi)
        print("\n--------------", role[1])
        
        for i in range(len(role[0])):
        
            if role[0][i][1] == condi:
                
                epo = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condi, role[0][i][0])).pick_channels(ch_to_keep, ordered=False)
                psds, freq = mne.time_frequency.psd_welch(epo, fmin=4.0, fmax=50.0, n_fft=int(epo.info['sfreq']))

                mean_epoch_count[condi][role[1]].append(psds.shape[0])

                all_psds[condi][role[1]].append(psds)

#%% 

# ############
# Plot PSDs for the whole spectra
# ############

avr_psd = deepcopy(all_psds)

for condi in ['ES', 'NS']:
    for role in ["SPEAKER", "LISTENER"]:
        print("Data in condition {} for {} is {} in average".format(condi, role, np.round(np.mean(mean_epoch_count[condi][role]), 3)))
        
        # average for each psd accross sensor / segments
        for i in range(len(avr_psd[condi][role])):
            avr_psd[condi][role][i] = np.mean(avr_psd[condi][role][i], axis=(0, 1))

        tmp = pd.DataFrame(list(zip(freq, np.array(avr_psd[condi][role]).sum(axis=0))), columns=['Freq','Power'])

        sns.lineplot(data=tmp, x="Freq", y="Power", label='{} - {}'.format(condi, role)).set(title='PSD per condition & role')
plt.legend(loc='upper right', title='Condition / Role')

# %%

# ############
# Plot PSDs for the beta band only
# ############


# Zoom into the beta band as defined here (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6633326/) 
mask_beta = (freq >= 16) & (freq <= 28) 
freq_beta = freq[mask_beta]

avr_psd_beta = deepcopy(all_psds)

for condi in ['ES', 'NS']:
    for role in ["SPEAKER", "LISTENER"]:
        
        # average for each psd accross sensor / segments
        for i in range(len(avr_psd_beta[condi][role])):
            avr_psd_beta[condi][role][i] = np.mean(avr_psd_beta[condi][role][i], axis=(0, 1))

        tmp = pd.DataFrame(list(zip(freq_beta, np.array(avr_psd_beta[condi][role]).sum(axis=0)[mask_beta])), columns=['Freq (Hz)','Power'])

        sns.lineplot(data=tmp, x="Freq (Hz)", y="Power", label='{} - {}'.format(condi, role)).set(title='PSD per condition & role (beta 16-28Hz)')
plt.legend(loc='upper right', title='Condition / Role')


#%%

# ############
# Compute the average PSDs for across condition & subject for each role 
# ############

fooof_psds = np.zeros((len(conditions), len(df_manifest['ES']), epo.info['nchan'], len(freq)))  # ES will be at index 0 and NS at index 1

for idx, condi in enumerate(conditions): 

    # loop over all SPEAKER subject
    for i in range(len(all_psds[condi]["SPEAKER"])):
        
        # Compute mean over the epoch dimension
        fooof_psds[idx][i] = np.mean(all_psds[condi]["SPEAKER"][i], axis=0)

# Compute the mean over all subjects
fooof_psds = fooof_psds.mean(axis=1)

# %%
# Plot topograpgy

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

#%%

# Initialize a FOOOFGroup object, with desired settings
fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose=False)

# Define the frequency range to fit
freq_range = [4.0, 50.0]

# Fit the power spectrum model across all channels
fg.fit(freq, fooof_psds[0] , freq_range)

# Check the overall results of the group fits
fg.plot()

#%%
# Define frequency bands of interest
bands = Bands({'theta': [4, 7],
               'alpha': [7, 14],
               'beta': [15, 30]})

# Extract alpha peaks
alphas = get_band_peak_fg(fg, bands.alpha)

# Extract the power values from the detected peaks
alpha_pw = alphas[:, 1]

# Plot the topography of alpha power
plot_topomap(alpha_pw, epo.info, cmap=cm.viridis, contours=0)

#%%

# Plot the topographies across different frequency bands
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, epo.info, cmap=cm.viridis, contours=0,
                         axes=axes[ind], show=False);

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})

#%% 
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Extracted and plot the power spectrum model with the most band power
    fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

    # Set some plot aesthetics & plot title
    axes[ind].yaxis.set_ticklabels([])
    axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})






