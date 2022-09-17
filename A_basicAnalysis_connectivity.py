#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
from collections import OrderedDict

from hypyp import analyses

import mne
import numpy as np

# Import custom tools
from utils import basicAnalysis_tools
from utils.useful_variable import *

mne.set_log_level('warning')
# matplotlib.use('Qt5Agg')

# Define paths and global variables
data_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/EEG_data_cleaned/"
save_path    = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"

# Here, choose where to save specific results without overwritting #######
# complex_sig  = "results_complex_signal/" # All 5 basic freq band
complex_sig  = "SELECTED_results_complex_signal/" # For specific freq band analysis

# ibc_result   = "results_ibc/"
ibc_result   = "SELECTED_results_ibc/"

c_value      = "results_cohensD/"
psds_result  = "results_psds/"


expConditionDir = ['SNS_ES_cleaned', 'SNS_NS_cleaned']
# expCondition = os.listdir(data_path)

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

# freq_bands = {'Theta': [4.0, 7.0],
#               'Alpha-Low': [7.5, 11.0],
#               'Alpha-High': [11.5, 13.0],
#               'Beta': [13.5, 29.5],
#               'Gamma': [30.0, 48.0]} 

freq_bands = OrderedDict(freq_bands)

# %% ####### Uncoment only to use the 'Selected' freq band #######
# ################################################################
to_pop_out = []

for i in freq_bands.keys():
    if i != 'Selected_freqBand':
        to_pop_out.append(i)
for i in to_pop_out:
    del freq_bands[i]
# ################################################################
# ################################################################

#%%
# Get analysis manifest
df_manifest = basicAnalysis_tools.get_analysis_manifest(data_path, expConditionDir, save_to=save_path)

#%% 
# Set analyse parameters
do_ibc          = True
do_psd          = True
do_inter_cohenD = False
do_intra_cohenD = False

ibc_metric = 'ccorr'
#  Supported connectivity measures
#   - 'envelope_corr': envelope correlation
#   - 'pow_corr': power correlation
#   - 'plv': phase locking value
#   - 'ccorr': circular correlation coefficient
#   - 'coh': coherence
#   - 'imaginary_coh': imaginary coherence

#%%
# Run Analysis
 
for condition in df_manifest.keys(): 
# for condition in ['ES']:
    print("- - > Doing condition {} ...".format(condition))

    for dyad in df_manifest[condition].keys():
    # for dyad in ['4']:

        print("- - - > Doing dyad {} ...".format(dyad))

        # Load participant's EEG #####################################################################
        epo1 = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condition, df_manifest[condition][dyad][0])).pick_channels(ch_to_keep, ordered=False)
        epo2 = mne.io.read_epochs_eeglab('{}SNS_{}_cleaned/{}'.format(data_path, condition, df_manifest[condition][dyad][1])).pick_channels(ch_to_keep, ordered=False)

        # Sanity check area ##########################################################################
        print("\nVerify equal epoch count: ")
        mne.epochs.equalize_epoch_counts([epo1, epo2])

        print("\nVerify equal channel count: ")
        ch_to_drop_in_epo1 = list(set(epo1.ch_names).difference(epo2.ch_names))
        ch_to_drop_in_epo2 = list(set(epo2.ch_names).difference(epo1.ch_names))
        if len(ch_to_drop_in_epo1) != 0:
            print('Dropping the following channel(s) in epo1: {}'.format(ch_to_drop_in_epo1))
            basicAnalysis_tools.add_reject_ch_manifest(condition=condition, dyad=dyad, sub_fname=df_manifest[condition][dyad][0], reject=ch_to_drop_in_epo1, save_to=save_path)
            epo1 = epo1.drop_channels(ch_to_drop_in_epo1)
        elif len(ch_to_drop_in_epo2) != 0:
            print('Dropping the following channel(s) in epo2: {}'.format(ch_to_drop_in_epo2))
            basicAnalysis_tools.add_reject_ch_manifest(condition=condition, dyad=dyad, sub_fname=df_manifest[condition][dyad][1], reject=ch_to_drop_in_epo2, save_to=save_path)
            epo2 = epo2.drop_channels(ch_to_drop_in_epo2)
        else:
            print('No channel to drop.')

        # Computing power spectrum ###################################################################
        if do_psd:
            for band in freq_bands.keys():
                print("psd for {} band".format(band))
                psd1 = analyses.pow(epo1, fmin=freq_bands[band][0], fmax=freq_bands[band][1], n_fft=1000, n_per_seg=1000, epochs_average=True)
                psd2 = analyses.pow(epo2, fmin=freq_bands[band][0], fmax=freq_bands[band][1], n_fft=1000, n_per_seg=1000, epochs_average=True)
                data_psd = np.array([psd1.psd, psd2.psd])
                np.save("{}{}dyad_{}_condition_{}_band_{}_psds.npy".format(save_path, psds_result, dyad, condition, band), data_psd, allow_pickle=True)

        # Computing IBC (init, analytic signal, and requency- and time-frequency-domain connectivity) ###################################################################
        if do_ibc:
            #  Initializing data and storage  #############################################################
            data_inter = np.array([])
            data_inter = np.array([epo1.get_data(), epo2.get_data()]) #, dtype=mne.io.eeglab.eeglab.EpochsEEGLAB) # Deprecation warning
            result_intra = []

            # Computing analytic signal per frequency band ################################################
            print("- - - - > Computing analytic signal per frequency band ...")
            sampling_rate = epo1.info['sfreq']
            complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands) # (2sub, n_epochs, n_channels, n_freq_bands, n_times)
            np.save("{}{}dyad_{}_condition_{}_complexsignal.npy".format(save_path, complex_sig, dyad, condition, complex_signal), complex_signal, allow_pickle=False)

            # Computing frequency- and time-frequency-domain connectivity ################################
            print("- - - - > Computing frequency- and time-frequency-domain connectivity ...")
            result = analyses.compute_sync(complex_signal, mode=ibc_metric, epochs_average=True) # (n_freq, 2*n_channels, 2*n_channels)
            np.save("{}{}dyad_{}_condition_{}_IBC_{}.npy".format(save_path, ibc_result, dyad, condition, ibc_metric), result, allow_pickle=False)

        # ################# INTER #####################################################
        if do_inter_cohenD:    
            # Slicing results to get the <<Inter-brain>> part of the matrix
            print("- - - - > Slicing results (inter) ...")
            n_ch = len(epo1.info['ch_names'])
            theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch:2*n_ch]

            # Choosing Alpha_Low for futher analyses for example
            values = alpha_low

            # Computing Cohens'D for further analyses for example #########################################
            print("- - - - > Computing Cohens'D ... ")
            C = (values - np.mean(values[:])) / np.std(values[:])
            np.save(save_path+c_value+"dyad_"+dyad+"_condition_"+condition+"_inter_cohenD.npy", C, allow_pickle=False)

        ################# INTRA #####################################################
        if do_intra_cohenD:
            # Slicing results to get the <<Intra-brain>> part of the matrix
            print("- - - - > Slicing results (intra) ...")
            for i in [0, 1]:
                theta, alpha_low, alpha_high, beta, gamma = result[:, (
                    i * n_ch):((i + 1) * n_ch), 
                    (i * n_ch): ((i + 1) * n_ch)]
                # choosing Alpha_Low for futher analyses for example
                values_intra = alpha_low
                values_intra -= np.diag(np.diag(values_intra))
                # computing Cohens'D for further analyses for example
                C_intra = (values_intra -
                        np.mean(values_intra[:])) / np.std(values_intra[:])
                # can also sample CSD values directly for statistical analyses
                result_intra.append(C_intra)
            
            np.save(save_path+c_value+"dyad_"+dyad+"_condition_"+condition+"_intra_cohenD.npy", result_intra, allow_pickle=True)

        #######################
        ##     if no ICA     ##   Un-comment the block bellow
        #######################

        ################# MVAR #####################################################
        # # Computing frequency- and time-frequency-domain connectivity measures obtained by MVARICA approach, based on MVAR models' coefficients. For instance: PDC measure, with MVAR model of order 2, extended infomax ICA method and checking the MVAR model stability.
        
        # mvar_result = analyses.compute_conn_mvar(complex_signal, 
        #                                         mvar_params={"mvar_order": 2, "fitting_method":"default", "delta": 0},
        #                                         ica_params={"method": "infomax_extended", "random_state": None},
        #                                         measure_params={"name": "pdc", "n_fft": 512}
        #                                         )
        # # Output: (1, frequency,channels, channels, n_fft)

        # no_ICA_result_intra = []
        # no_ICA_result_inter = []

        # # Slicing results to get the INTER-brain of the connectivity matrix and assigning the maximum value in the frequency spectrum (mvar-based connectivity measures are calculated over a frequency range assigned by n_fft variable, here n_fft = 512)
        # for i in [0, 1]:
        #     mvar_result = mvar_result.squeeze()
        #     if i == 0 :
        #         mvar_theta, mvar_alpha_low, mvar_alpha_high, mvar_beta, mvar_gamma =  mvar_result[:, n_ch:n_ch*2, 0:n_ch, :]
        #     else:
        #         mvar_theta, mvar_alpha_low, mvar_alpha_high, mvar_beta, mvar_gamma =  mvar_result[:, 0:n_ch, n_ch:n_ch*2, :]
        #     # choosing Alpha_Low for futher analyses for example
        #     auxiliary = np.zeros((n_ch, n_ch), dtype=mvar_result.dtype)
        #     for j in range(0, n_ch):
        #         for k in range(0, n_ch):
        #             auxiliary[j, k] = np.amax(mvar_alpha_low[j,k])
        #     mvar_values_inter = auxiliary
        #     # computing Cohens'D for further analyses for example
        #     mvar_C_inter = (mvar_values_inter -
        #             np.mean(mvar_values_inter[:])) / np.std(mvar_values_inter[:])
        #     # can also sample CSD values directly for statistical analyses
        #     no_ICA_result_inter.append(mvar_C_inter)

        # # And now slicing for the INTRA-brain of the connectivity matrix ...
        # for i in [0, 1]:
        #     mvar_result = mvar_result.squeeze()
        #     mvar_theta, mvar_alpha_low, mvar_alpha_high, mvar_beta, mvar_gamma =  mvar_result[:, i*n_ch:n_ch*(i+1), i*n_ch:n_ch*(i+1), :]
        #     # choosing Alpha_Low for futher analyses for example
        #     auxiliary = np.zeros((n_ch, n_ch), dtype=mvar_result.dtype)
        #     for j in range(0, n_ch):
        #         for k in range(0, n_ch):
        #             auxiliary[j, k] = np.amax(mvar_alpha_low[j, k])
        #     mvar_alpha_low = auxiliary
        #     mvar_values_intra = mvar_alpha_low
        #     mvar_values_intra -= np.diag(np.diag(mvar_values_intra))
        #     # computing Cohens'D for further analyses for example
        #     mvar_C_intra = (mvar_values_intra -
        #             np.mean(mvar_values_intra[:])) / np.std(mvar_values_intra[:])
        #     # can also sample CSD values directly for statistical analyses
        #     no_ICA_result_intra.append(mvar_C_intra)

        # # STATISTICS ################################################################################
        # # <<<1>>> MNE test without any correction
        # psd1_mean = np.mean(psd1.psd, axis=1)
        # psd2_mean = np.mean(psd2.psd, axis=1)
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


        print('- - > Done.')

# %%
