#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

# %%##########
# Import packages, paths, and useful variables.
# ############
from copy import deepcopy

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

import json

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
roles           = ['SPEAKER', 'LISTENER']
# selected_freqB  = [16, 28.0]
selected_freqB  = [8.0, 12.0]
#%% 

df_manifest = basicAnalysis_tools.get_analysis_manifest(data_path, expConditionDir, save_to=None)

# %%##########
# Make list of the Speaker and Listener
# ############
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

input('sure you wanna do that? (if not, you can still interrupt the process)')

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


# Save the computed psd manifest NOT WORKING BC TypeError: Object of type ndarray is not JSON serializable
# Could "jsonify" np.arrays using the ".tolist()" method 

# json_string = json.dumps(all_psds)
# f = open(save_path+"PSDS_RESULTS_DICT.json","w")
# f.write(json_string)
# f.close()

# %%##########
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

del avr_psd

# %%##########
# Plot PSDs for the selected band only
# ############


# Zoom into the beta band as defined here (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6633326/) 
mask_beta = (freq >= selected_freqB[0]) & (freq <= selected_freqB[1]) 
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

del tmp

#%%
#           ###########   2. Plot topography    ###########

# ############
# Compute the average PSDs for across condition & subject for each role 
# ############

# fooof_psds = np.zeros((len(conditions), len(df_manifest['ES']), epo.info['nchan'], len(freq)))  # ES will be at index 0 and NS at index 1


# for idx, condi in enumerate(conditions): 

#     # loop over all SPEAKER subject
#     for i in range(len(all_psds[condi]["SPEAKER"])):
        
#         # Compute mean over the epoch dimension
#         fooof_psds[idx][i] = np.mean(all_psds[condi]["SPEAKER"][i], axis=0)

# # Compute the mean over all subjects
# fooof_psds = fooof_psds.mean(axis=1)




#%%
# Initialize fooof df with (roles, condis , sensors, freqs)

all_fooof_psds = np.zeros((len(roles), len(conditions), len(df_manifest['ES']), epo.info['nchan'], len(freq)))  # ES will be at index 0 and NS at index 1


for role_idx, role in enumerate(roles):
    for idx, condi in enumerate(conditions): 
        for i in range(len(all_psds[condi][role])):
            
            if all_psds[condi][role][i].shape[1] != 22:
                pass
            else:
                # Compute mean over the epoch dimension
                all_fooof_psds[role_idx][idx][i] = np.mean(all_psds[condi][role][i], axis=0)

#########
# The shape of np.arr fooof_psds is now \
#               (2, 2, 36, 22, 47) 
#               (role [SPEAKER, LISTENER], condi [ES, NS], subjects, sensors, freqs)
#########

#%%
# Compute the mean over all subjects
all_fooof_psds = all_fooof_psds.mean(axis=2)

#%% 
# Plot all averaged PSD

sns.lineplot(
    data=pd.DataFrame(list(zip(
        freq, 
        all_fooof_psds.mean(axis=(0, 1, 2)))), columns=['Freq (Hz)','Power']), 

    x="Freq (Hz)", y="Power", 
    label='PSD (all conditions / all sensors / all roles)').set(
        title='PSD (all conditions / all sensors / all roles)')

# %%##########
# Define NaNs policy, Initialize FOOOFGroup w/ desired settings, and filter psd df
# ############

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

fg = FOOOFGroup(
    peak_width_limits=[0.5, 10],
    min_peak_height=0.15,
    peak_threshold=1.,
    # max_n_peaks=20, 
    verbose=True)

# Define the frequency range to fit
freq_range = [np.min(freq), np.max(freq)]

# Select which role to look into  
looking_into_role = 'listener'
if looking_into_role == 'both roles':
    fooof_psds = all_fooof_psds.mean(0)
elif looking_into_role == 'speaker':
    fooof_psds = all_fooof_psds[0]
elif looking_into_role == 'listener':
    fooof_psds = all_fooof_psds[1]
else:
    print('Not understood.')

# Fit the power spectrum model across all channels in condition of choice
looking_into_condi = 'ES'
if looking_into_condi == 'both conditions':
    fg.fit(freq, fooof_psds.mean(axis=0) , freq_range) # To fit all condi in SPEAKER
elif looking_into_condi == 'ES':
    fg.fit(freq, fooof_psds[0] , freq_range) # To fit ES
elif looking_into_condi == 'NS':
    fg.fit(freq, fooof_psds[1] , freq_range) # To fit NS
else:
    print('Not understood.')

# Check the overall results of the group fits
fg.plot()
fg.print_results()

# %%##########
# Define frequency bands of interest
# ############

bands = Bands({'theta': [4, 7],
               'alpha': [7, 14],
               'beta': [15, 30]})

bands_beta_1 = Bands({
    'beta_1': [16, 18],
    'beta_2': [18, 20],
    'beta_3': [20, 22]})

bands_beta_2 = Bands({
    'beta_4': [22, 24],
    'beta_5': [24, 26],
    'beta_6': [26, 28]})

bands_alpha = Bands({
    'alpha_1': [8, 10],
    'alpha_2': [10, 12],
    'alpha_all': [8, 12]
    })

# unique_band = Bands({'beta':[15, 20], 'beta_2':[15, 20]})

# # Extract alpha peaks
# betas = get_band_peak_fg(fg, bands.beta)

# # Extract the power values from the detected peaks
# beta_pw = betas[:, 1]

# # Plot the topography of alpha power
# mne.viz.plot_topomap(beta_pw, epo.info, cmap=cm.viridis, contours=0)

# %%##########
# Plot the topographies across different frequency bands
# ############
# for bbandd in [bands_beta_1, bands_beta_2]:
for bbandd in [bands_alpha]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Topographic map for beta power\nfor {} in {}'.format(
        looking_into_role, looking_into_condi), 
        y=1.15, size=22, weight='regular')
    for ind, (label, band_def) in enumerate(bbandd):

        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

        # Create a topomap for the current oscillation band
        mne.viz.plot_topomap(band_power, epo.info, cmap=cm.viridis, contours=6,
                            axes=axes[ind], show=False, outlines='skirt', res=50,
                            sensors='r+', names=epo.info['ch_names'], show_names=True)

        # Set the plot title
        axes[ind].set_title('[{}-{}] Hz'.format(str(band_def[0]), str(band_def[1])), {'fontsize' : 20})

plt.savefig(save_path + 'plots/topographic_freq_domain/' + 'rename.pdf')
# %%##########
# Plot model fit
# ############

for bbandd in [bands_beta_1, bands_beta_2]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ind, (label, band_def) in enumerate(bbandd):

        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

        # Extracted and plot the power spectrum model with the most band power
        fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

        # Set some plot aesthetics & plot title
        axes[ind].yaxis.set_ticklabels([])
        axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})

# %%##########
# Plot head model
# ############

epo.plot_sensors(kind='topomap', show_names=True)

# %%
