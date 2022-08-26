#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
# Import packages, paths, and useful variables.
from hypyp import stats
from hypyp import viz
from hypyp import analyses
import json
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy

# Import custom tools
from utils import basicAnalysis_tools
from utils.useful_variable import *

#%% 
# Load data

data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/EEG_data_cleaned/"
save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"

psds_result_allFreq  = "results_psds_allFreq/"

expConditionDir = ['SNS_ES_cleaned', 'SNS_NS_cleaned']

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

all_psds = {
    'ES':{
        "SPEAKER":[], 
        "LISTENER":[]}, 
    'NS':{
        "SPEAKER":[], 
        "LISTENER":[]}}

mean_epoch_count = []

for condi in ['ES', 'NS']:
  
    for role in [(speaker_list, "SPEAKER"), (listener_list, "LISTENER")]:
        
        print("\n-------", condi)
        print("\n--------------", role[1])
        
        for i in range(len(role[0])):
        
            if role[0][i][1] == condi:
                
                print(role[0][i][0])
                
                epo = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condi, role[0][i][0])).pick_channels(ch_to_keep, ordered=False)
                psds, freq = mne.time_frequency.psd_welch(epo)
                
                print("\t --> SHAPE: ", psds.shape)
                mean_epoch_count.append(psds.shape[0])

                all_psds[condi][role[1]].append(psds)

#%% 
np.mean(mean_epoch_count) 

#%% 
# Now compute the mean of all these psds per condi / role
np.mean( np.array([ old_set, new_set ]), axis=0 )

#%%

#         # Load participant's EEG #####################################################################
#         epo1 = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condition, df_manifest[condition][dyad][0])).pick_channels(ch_to_keep, ordered=False)
#         epo2 = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condition, df_manifest[condition][dyad][1])).pick_channels(ch_to_keep, ordered=False)

#         # Sanity check area ##########################################################################
#         print("\nVerify equal epoch count: ")
#         mne.epochs.equalize_epoch_counts([epo1, epo2])

#         print("\nVerify equal channel count: ")
#         ch_to_drop_in_epo1 = list(set(epo1.ch_names).difference(epo2.ch_names))
#         ch_to_drop_in_epo2 = list(set(epo2.ch_names).difference(epo1.ch_names))
#         if len(ch_to_drop_in_epo1) != 0:
#             print('Dropping the following channel(s) in epo1: {}'.format(ch_to_drop_in_epo1))
#             basicAnalysis_tools.add_reject_ch_manifest(condition=condition, dyad=dyad, sub_fname=df_manifest[condition][dyad][0], reject=ch_to_drop_in_epo1, save_to=save_path)
#             epo1 = epo1.drop_channels(ch_to_drop_in_epo1)
#         elif len(ch_to_drop_in_epo2) != 0:
#             print('Dropping the following channel(s) in epo2: {}'.format(ch_to_drop_in_epo2))
#             basicAnalysis_tools.add_reject_ch_manifest(condition=condition, dyad=dyad, sub_fname=df_manifest[condition][dyad][1], reject=ch_to_drop_in_epo2, save_to=save_path)
#             epo2 = epo2.drop_channels(ch_to_drop_in_epo2)
#         else:
#             print('No channel to drop.')

#         # Computing power spectrum ###################################################################
#         if do_psd:
#             for band in freq_bands.keys():
#                 print("psd for {} band".format(band))
#                 psd1 = analyses.pow(epo1, fmin=freq_bands[band][0], fmax=freq_bands[band][1], n_fft=1000, n_per_seg=1000, epochs_average=True)
#                 psd2 = analyses.pow(epo2, fmin=freq_bands[band][0], fmax=freq_bands[band][1], n_fft=1000, n_per_seg=1000, epochs_average=True)
#                 data_psd = np.array([psd1.psd, psd2.psd])
#                 np.save("{}{}dyad_{}_condition_{}_band_{}_psds.npy".format(save_path, psds_result, dyad, condition, band), data_psd, allow_pickle=True)

# %%
