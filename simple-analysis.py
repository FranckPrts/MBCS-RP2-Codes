#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
from collections import OrderedDict
from hypyp import analyses
import mne
import numpy as np
# Import custom tools
from utils import IBC_analysis

# Define paths and global variables
data_path = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/EEG_data_cleaned/"
save_path = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"

expCondition = ['SNS_ES_cleaned', 'SNS_NS_cleaned']
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

freq_bands = {'Theta': [4, 7],
              'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13],
              'Beta': [13.5, 29.5],
              'Gamma': [30, 48]}
freq_bands = OrderedDict(freq_bands)

# Get analysis manifest
df_manifest = IBC_analysis.get_analysis_manifest(data_path, expCondition)

#%% 
# for condition in df_manifest.keys(): TODO
for condition in ['SNS_ES_cleaned']:
    print("- - > Doing condition {} ...".format(condition))

    # for dyad in df_manifest[condition].keys(): TODO
    for dyad in ['21']:
        print("- - - > Doing dyad {} ...".format(dyad))

        epo1 = mne.io.read_epochs_eeglab(data_path + condition +'/'+ df_manifest[condition][dyad][0]).pick_channels(ch_to_keep, ordered=False)
        epo2 = mne.io.read_epochs_eeglab(data_path + condition +'/'+ df_manifest[condition][dyad][1]).pick_channels(ch_to_keep, ordered=False)

        print("\nChecking amount of epoch available for each subject:")
        mne.epochs.equalize_epoch_counts([epo1, epo2])

        # Computing power spectrum
        # psd1 = analyses.pow(epo1, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True)
        # psd2 = analyses.pow(epo2, fmin=7.5, fmax=11, n_fft=1000, n_per_seg=1000, epochs_average=True)
        # data_psd = np.array([psd1.psd, psd2.psd])

        #  Initializing data and storage
        data_inter = np.array([epo1, epo2])
        result_intra = []

        # Computing analytic signal per frequency band
        print("- - - - > Computing analytic signal per frequency band ...")
        sampling_rate = epo1.info['sfreq']
        complex_signal = analyses.compute_freq_bands(data_inter, sampling_rate, freq_bands)

        # Computing frequency- and time-frequency-domain connectivity, 'ccorr' for example
        print("- - - - > Computing frequency- and time-frequency-domain connectivity ...")
        result = analyses.compute_sync(complex_signal, mode='ccorr')

        # Slicing results to get the Inter-brain part of the matrix
        print("- - - - > Slicing results ...")
        n_ch = len(epo1.info['ch_names'])
        theta, alpha_low, alpha_high, beta, gamma = result[:, 0:n_ch, n_ch:2*n_ch]

        # Choosing Alpha_Low for futher analyses for example
        values = alpha_low

        # Computing Cohens'D for further analyses for example
        print("- - - - > Computing Cohens'D ... ")
        C = (values - np.mean(values[:])) / np.std(values[:])



        print('- - > Done.')

# %%



