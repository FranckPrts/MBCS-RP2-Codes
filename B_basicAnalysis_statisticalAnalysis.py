#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
from hypyp import stats
from hypyp import viz
from hypyp import analyses
import json
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

# Import custom tools
from utils import basicAnalysis_tools
from utils.useful_variable import *

mne.set_log_level('warning')
# matplotlib.use('Qt5Agg')

eeg_sampl     = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/samples/sample_epochs.set"
mani_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"
fig_save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots/"
ibc_data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/results_ibc/"
psd_data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/results_psds/"
# save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots"

f=open(mani_path+"df_manifest.json")
df_manifest = json.load(f)
f.close()

ibc_metrics = ['envelope_corr', 'pow_corr', 'plv', 'ccorr', 'coh', 'imaginary_coh']
conditions  = list(df_manifest.keys())
dyads       = list(set(df_manifest[conditions[1]]).intersection(df_manifest[conditions[0]]))
bands       = freq_bands.keys()

#  TODO make a func that deletes the dyad that have to be rejected
dyads.remove('36') # removing dyad 

#%% Loading data 
epo1 = mne.io.read_epochs_eeglab(eeg_sampl).pick_channels(ch_to_keep, ordered=False)
n_ch = len(epo1.info['ch_names'])

#%% 
#  Output is shape (freq_banddyads, sensor(x2), sensor(x2))
ibc_df, rejected_dyad = basicAnalysis_tools.create_ibc_manifest(
    data_path    = ibc_data_path, 
    mani_path    = mani_path, # DOES'T WORK (yet) bc np.arr are multi-dim
    conditions   = conditions,
    ibc_metrics  = ['ccorr'],
    n_ch         = n_ch,
    nb_freq_band = len(freq_bands.keys()), 
    # specific_file ='dyad_16_condition_ES_IBC_ccorr.npy',
    save         = True
    )

#%%
# Prints qverqge IBC measure per condition x frequency band
for band in bands:
    for condi in conditions :
        #  Slice the INTER part of the matrice for the frequency band of choice
        conVal = ibc_df[condi]["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch]
        # Compute mean over all sensor
        print(condi, band, conVal.mean())

#%%  Look into the IBC value distribution across ONE freauency band
band = 'Alpha-Low'
conVal_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))
g = sns.displot(conVal_ES, kind="kde")
plt.title(label='Distribution of IBC vqlues in {}'.format(band), loc='left') 
# plt.savefig('{}{}.png'.format(fig_save_path, 'fig1'))
plt.show(g)
#%% Compute mean connectivity measure on all sensors while keeping freqband and sub dimension
conVal_freqSub = ibc_df['ES']["ccorr"][:, :, 0:n_ch, n_ch:2*n_ch].mean(axis=(2, 3))
g = sns.displot(conVal_freqSub.transpose(), kind="kde", legend = False)
plt.title(label='Average connectiviy over all sensors', loc='left') 
plt.legend(title='Frequency bands', loc='upper right', labels=list(freq_bands_ord.keys())[::-1])
plt.show(g)

#%%
for band in bands:
    conVal_freqSub_NS = ibc_df['NS']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))
    conVal_freqSub_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))

    print("\n"+band)# Print band then average value
    print("Mean IBC in NS: {}".format(ibc_df['NS']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean())) 
    print("Mean IBC in ES: {}".format(ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean())) 
    
    print(scipy.stats.shapiro(conVal_freqSub_NS))
    print(scipy.stats.shapiro(conVal_freqSub_ES))

    # Conduct Paired Sample T-Test on IBC 
    print(scipy.stats.ttest_rel(conVal_freqSub_NS, conVal_freqSub_ES))


#%% 
# theta, alpha_low, alpha_high, beta, gamma = ibc_df["ES"]["ccorr"][:, :, 0:n_ch, n_ch:2*n_ch]

# We're now looking into 
# values = alpha_low

# print("- - - - > Computing Cohens'D ... ")
# C = (values - np.mean(values[:])) / np.std(values[:])
# viz.viz_2D_topomap_inter(epo1, epo1, C, threshold='auto', steps=10, lab=True)

#%% Prepare metadata for statcondCluster 
# define the frequency band of interest 
fr_b = "Alpha-Low"

# Create the freq_list varialbe (e.g., for Alpha-Low [7.5, 11.0]: [7.5, 8, 8.5, ..., 11.])
freq_list = np.linspace(
    start=float(freq_bands[fr_b][0]), 
    stop=float(freq_bands[fr_b][1]), 
    endpoint=True,
    num=int((freq_bands[fr_b][1]-freq_bands[fr_b][0])/0.5+1))

con_matrixTuple = stats.con_matrix(epo1, freqs_mean=freq_list)
ch_con_freq = con_matrixTuple.ch_con_freq

#%% Create and populate data_group containing the PSD value of all participants in one list of 2 list for each conditions
data_group = [np.zeros([len(dyads)*2, n_ch, len(freq_list)], dtype=np.float32), 
              np.zeros([len(dyads)*2, n_ch, len(freq_list)], dtype=np.float32)]

for condi in conditions:
    counter = 0
    for dyad in dyads:
        # load the psd data
        psd1, psd2 = np.load("{}dyad_{}_condition_{}_psds.npy".format(psd_data_path, dyad, condi))
        
        # To recreate the named tuple hypyp.analysis.pow() initialy makes
        # psd_tuple = namedtuple('PSD', ['freq_list', 'psd'])
        # psd1 = psd_tuple(freq_list=freq_list, psd=psd)

        print("Dyad #{}".format(dyad))
        if condi == 'ES':
            data_group[0][counter] = psd1
            data_group[0][counter+1] = psd2
        elif condi == 'NS':
            data_group[1][counter] = psd1
            data_group[1][counter+1] = psd2
        else:
            print("something went wrong with the condition: ", condi, dyad)
        # print(condi, '–––', dyad, '–––', psd1.mean(), psd2.mean())
        counter+=2


#%% 
statscondCluster = stats.statscondCluster(data=data_group,
                                          freqs_mean=freq_list,
                                          ch_con_freq=scipy.sparse.bsr_matrix(ch_con_freq),
                                          tail=0,
                                          n_permutations=5000,
                                          alpha=0.05)

F_obs, clusters, cluster_pv, H0, F_obs_plot = statscondCluster

#%% 
analyses.indices_connectivity_interbrain()

#%% 
viz.plot_significant_sensors(F_obs, epo1)

#%% 
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