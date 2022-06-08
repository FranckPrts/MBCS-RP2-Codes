#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
from hypyp import stats
from hypyp import viz
import json
import matplotlib
import mne
import numpy as np
import scipy
# Import custom tools
from utils import basicAnalysis_tools

mne.set_log_level('warning')
# matplotlib.use('Qt5Agg')

eeg_sampl     = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/samples/sample_epochs.set"
data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"
save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots"

ch_to_keep= [
    'Fp1', 'Fp2', 'F3', 'F4', 
    'C3', 'C4', 
    'P3', 'P4', 
    'O1', 'O2', 
    'F7', 'F8', 
    'T7', 'T8', 
    'P7', 'P8', 
    'Fz', 'Cz', 'Pz', 
    'AFz', 'CPz', 'POz']

#%% Loading data 

f=open(data_path+"df_manifest.json")
df_manifest = json.load(f)
f.close()

condition  = list(df_manifest.keys())
avail_dyad = list(set(df_manifest[condition[1]]).intersection(df_manifest[condition[0]]))

epo1 = mne.io.read_epochs_eeglab(eeg_sampl).pick_channels(ch_to_keep, ordered=False)

###################### TMP
conditions = 'NS'
dyad = '13'
###################### TMP

psd1, psd2 = np.load("{}results_psds/dyad_{}_condition_{}_psds.npy".format(data_path, dyad, conditions))

#%% 
# STATISTICS ################################################################################
# <<<1>>> MNE test without any correction
psd1_mean = np.mean(psd1, axis=1)
psd2_mean = np.mean(psd2, axis=1)
X = np.array([psd1_mean, psd2_mean])

T_obs, p_values, H0 = mne.stats.permutation_t_test(X=X, n_permutations=5000,
                                                tail=0, n_jobs=1)

# <<<2>>> Computes statistical t test on participant measure (e.g. PSD) for a condition.
statsCondTuple = stats.statsCond(data=np.array([psd1, psd2]),
                                epochs=epo1,
                                n_permutations=5000,
                                alpha=0.05)
                                # T_obs, p_values, H0, adj_p, T_obs_plot
# viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs_plot, epochs=epo1)

# <<<3>>> Non-parametric cluster-based permutations 
# Creating matrix of a priori connectivity between channels across space and frequencies based on their position, in the Alpha_Low band for example
con_matrixTuple = stats.con_matrix(epo1, freqs_mean=psd1.freq_list)
ch_con_freq = con_matrixTuple.ch_con_freq

# Creating two fake groups with twice the 'participant1' and twice the 'participant2'
data_group = [np.array([psd1.psd, psd1.psd]), np.array([psd2.psd, psd2.psd])]

statscondCluster = stats.statscondCluster(data=data_group,
                                        freqs_mean=psd1.freq_list,
                                        ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                        tail=0,
                                        n_permutations=5000,
                                        alpha=0.05)









# %%
# viz.viz_2D_topomap_inter(epo1, epo2, C, threshold='auto', steps=10, lab=True)

#%%
# viz.viz_3D_inter(epo1, epo2, C, threshold='auto', steps=10, lab=False)