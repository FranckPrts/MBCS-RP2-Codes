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
from utils.useful_variable import *

mne.set_log_level('warning')
# matplotlib.use('Qt5Agg')

eeg_sampl     = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/samples/sample_epochs.set"
mani_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"
data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/results_ibc/"
# save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots"

f=open(mani_path+"df_manifest.json")
df_manifest = json.load(f)
f.close()

conditions = list(df_manifest.keys())
dyads     = list(set(df_manifest[conditions[1]]).intersection(df_manifest[conditions[0]]))
ibc_metrics = ['envelope_corr', 'pow_corr', 'plv', 'ccorr', 'coh', 'imaginary_coh']

#%% Loading data 

epo1 = mne.io.read_epochs_eeglab(eeg_sampl).pick_channels(ch_to_keep, ordered=False)
n_ch = len(epo1.info['ch_names'])

###################### TMP
# conditions = ['NS']
dyads = ['1', '2']
tmp = {
    "ES":{
        '1':np.array([[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]]]), 
        '2':np.array([[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]]])}, 
    "NS":{
        '1':np.array([[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]]]), 
        '2':np.array([[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]]])}}
###################### TMP

#%% 

ibc_df = basicAnalysis_tools.create_ibc_manifest(
    data_path    = data_path, 
    mani_path    = mani_path, # DOES'T WORK (yet) bc np.arr are multi-dim
    conditions   = conditions,
    ibc_metrics  = ['ccorr'],
    n_ch         = n_ch,
    nb_freq_band = len(freq_bands.keys()), 
    # specific_file ='dyad_16_condition_ES_IBC_ccorr.npy',
    save         = True
    )
#%% 
theta, alpha_low, alpha_high, beta, gamma = ibc_df[:, 0:n_ch, n_ch:2*n_ch]

values = alpha_low

print("- - - - > Computing Cohens'D ... ")
C = (values - np.mean(values[:])) / np.std(values[:])

viz.viz_2D_topomap_inter(epo1, epo1, C, threshold='auto', steps=10, lab=True)

# for condi in conditions:
#     for dyad in dyads:
#         print(tmp[condi][dyad])

#%% 


# psd1, psd2 = np.load("{}results_psds/dyad_{}_condition_{}_psds.npy".format(data_path, dyad, conditions))

# #%% 
# # STATISTICS ################################################################################
# # <<<1>>> MNE test without any correction
# psd1_mean = np.mean(psd1, axis=1)
# psd2_mean = np.mean(psd2, axis=1)
# X = np.array([psd1_mean, psd2_mean])

# T_obs, p_values, H0 = mne.stats.permutation_t_test(X=X, n_permutations=5000,
#                                                 tail=0, n_jobs=1)

# # <<<2>>> Computes statistical t test on participant measure (e.g. PSD) for a condition.
# statsCondTuple = stats.statsCond(data=np.array([psd1, psd2]),
#                                 epochs=epo1,
#                                 n_permutations=5000,
#                                 alpha=0.05)
#                                 # T_obs, p_values, H0, adj_p, T_obs_plot
# # viz.plot_significant_sensors(T_obs_plot=statsCondTuple.T_obs_plot, epochs=epo1)

# # <<<3>>> Non-parametric cluster-based permutations 
# # Creating matrix of a priori connectivity between channels across space and frequencies based on their position, in the Alpha_Low band for example
# con_matrixTuple = stats.con_matrix(epo1, freqs_mean=psd1.freq_list)
# ch_con_freq = con_matrixTuple.ch_con_freq

# # Creating two fake groups with twice the 'participant1' and twice the 'participant2'
# data_group = [np.array([psd1.psd, psd1.psd]), np.array([psd2.psd, psd2.psd])]

# statscondCluster = stats.statscondCluster(data=data_group,
#                                         freqs_mean=psd1.freq_list,
#                                         ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
#                                         tail=0,
#                                         n_permutations=5000,
#                                         alpha=0.05)









# %%
# viz.viz_2D_topomap_inter(epo1, epo2, C, threshold='auto', steps=10, lab=True)

#%%
# viz.viz_3D_inter(epo1, epo2, C, threshold='auto', steps=10, lab=False)