#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
# Import packages, paths, and useful variables.
from copy import deepcopy
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

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

# Create simila dict structure to store amount of epochs per subject
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

tt = deepcopy(all_psds)

for condi in ['ES', 'NS']:
    for role in ["SPEAKER", "LISTENER"]:
        print("Data in condition {} for {} is {} in average".format(condi, role, np.round(np.mean(mean_epoch_count[condi][role]), 3)))
        
        # average for each psd accross sensor / segments
        for i in range(len(tt[condi][role])):
            tt[condi][role][i] = np.mean(tt[condi][role][i], axis=(0, 1))

        tmp = pd.DataFrame(list(zip(freq, np.array(tt[condi][role]).sum(axis=0))), columns=['Freq','Power'])

        sns.lineplot(data=tmp, x="Freq", y="Power", label='{} - {}'.format(condi, role)).set(title='PSD per condition & role')
plt.legend(loc='upper right', title='Condition / Role')

# %%
