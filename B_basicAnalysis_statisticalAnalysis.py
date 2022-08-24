#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%% 
# Import packages, paths, and useful variables.
#  from cgi import print_arguments
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

eeg_sampl       = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/samples/sample_epochs.set"
mani_path       = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/"
fig_save_path   = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots/"
ibc_data_path   = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/results_ibc/"
psd_data_path   = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/results_psds/"

save_format = 'pdf'

f=open(mani_path+"df_manifest.json")
df_manifest = json.load(f)
f.close()

ibc_metrics = ['envelope_corr', 'pow_corr', 'plv', 'ccorr', 'coh', 'imaginary_coh']
conditions  = list(df_manifest.keys())
dyads       = list(set(df_manifest[conditions[1]]).intersection(df_manifest[conditions[0]]))
bands       = freq_bands.keys()

# Removing rejected dyad(s) 
dyads.remove('36') 

#%% 
# Loading an epoch file to get useful metadata 
epo1 = mne.io.read_epochs_eeglab(eeg_sampl).pick_channels(ch_to_keep, ordered=False)
n_ch = len(epo1.info['ch_names'])

#%% 
# #### Create the IBC manifest ####
#  Output is shape (freq_banddyads, sensors(x2), sensors(x2))
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


#           ###########   0. averages    ###########

#%% 
# ############ 0.1
# Prints average IBC measure per condition x frequency band
# ############

for band in bands:
    for condi in conditions :
        #  Slice the INTER part of the matrice for each frequency band
        conVal = ibc_df[condi]["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch]
        # Compute mean over all sensor
        print(condi, band, conVal.mean())


#%% 
# ############ 0.2
# Prints average IBC measure per condition x frequency band x ROIs
# ############

for band in bands: 
    for condi in conditions :
        for roi in ROIs.keys():

            cut = basicAnalysis_tools.get_ch_idx(roi=roi, n_ch=n_ch, quadrant='inter')
            
            conVal = ibc_df[condi]["ccorr"][fqb2idx[band]][:, cut[0], cut[1]]
            # conVal = ibc_df[condi]["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch] # archive
            
            # Compute mean over selected sensors
            print(condi, band, roi, conVal.mean())


#           ###########   1. plots    ###########

save_plot = False

#%%
# ############ 1.1
# IBC distribution across frequency band
# ############

for band in bands:
    conVal_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))
    g = sns.displot(conVal_ES, kind="kde")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('IBC value distribution in {} (all sensors)'.format(band), loc='left')
    if save_plot:
        plt.savefig('{}IBC_distrib/all_ch/IBC_distrib_{}_all_chans.{}'.format(fig_save_path, band, save_format))

#%%
# ############ 1.2
# IBC distribution across frequency band x ROIs
# ############

for band in bands:
    for roi in ROIs.keys():
        cut = basicAnalysis_tools.get_ch_idx(roi=roi, n_ch=n_ch, quadrant='inter')
        conVal_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:,  cut[0], cut[1]].mean(axis=(1, 2))
        g = sns.displot(conVal_ES, kind="kde")
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title('IBC value distribution in {} ({})'.format(band, roi), loc='left')
        if save_plot:
            plt.savefig('{}IBC_distrib/per_band/per_roi/{}/IBC_distrib_{}_{}.{}'.format(fig_save_path, roi, band, roi, save_format))


#%% 
# ############ 1.3
# Average IBC per frequency band, all sensors
# ############
conVal_freqSub = ibc_df['ES']["ccorr"][:, :, 0:n_ch, n_ch:2*n_ch].mean(axis=(2, 3))
g = sns.displot(conVal_freqSub.transpose(), kind="kde", legend = False)
g.fig.subplots_adjust(top=.95)
g.ax.set_title(label='IBC value distrib per frequency band (all sensors)', loc='left') 
plt.legend(title='Frequency bands', loc='upper right', labels=list(freq_bands_ord.keys())[::-1])
if save_plot:
    plt.savefig('{}IBC_distrib/IBC_distrib_allFreqband_allChans.{}'.format(fig_save_path, save_format))

#%% 
# ############ 1.4
# Average IBC per frequency band, all sensors, per condition
# ############

tmp=[]
for condi in conditions:
    tmp_long = pd.DataFrame(ibc_df[condi]["ccorr"][:, :, 0:n_ch, n_ch:2*n_ch].mean(axis=(2, 3)).T)
    tmp_long.columns = list(freq_bands_ord.keys())
    tmp_long['condition']=condi
    tmp.append(tmp_long)

ibc_5freq_2condi = pd.concat(tmp, ignore_index=True)

ibc_5freq_2condi_long = pd.melt(ibc_5freq_2condi, id_vars=['condition'],var_name='FrequencyBand', value_name='ibc')

g = sns.FacetGrid(ibc_5freq_2condi_long, col="condition", hue="FrequencyBand")
g.map(sns.histplot, "ibc")#, hue='FrequencyBand')
if save_plot:
    plt.savefig('{}IBC_distrib/all_ch/IBC_distrib_allFreqband_allChans_perCondition.{}'.format(fig_save_path, save_format))

#%% 
# ############ 1.5
# Average IBC per frequency band, per condition, per ROI
# ############

tmp=[]
for roi in ROIs.keys():
    for condi in conditions:

        # Get (proper) indexing of the sensor for ROI
        cut = basicAnalysis_tools.get_ch_idx(roi=roi, n_ch=n_ch, quadrant='inter')

        tmp_long = pd.DataFrame(ibc_df[condi]["ccorr"][:, :, cut[0], cut[1]].mean(axis=(2, 3)).T)
        tmp_long.columns = list(freq_bands_ord.keys())
        tmp_long['condition']=condi
        tmp.append(tmp_long)

    ibc_5freq_2condi = pd.concat(tmp, ignore_index=True)
    ibc_5freq_2condi_long = pd.melt(ibc_5freq_2condi, id_vars=['condition'],var_name='FrequencyBand', value_name='ibc')

    g = sns.FacetGrid(ibc_5freq_2condi_long, col="condition", hue="FrequencyBand")
    g.map(sns.histplot, "ibc")#, hue='FrequencyBand')
    # g.axes.set_title(label='IBC value distrib ', loc='left') 
if save_plot:
    plt.savefig('{}IBC_distrib/per_roi/IBC_distrib_allFreqband_{}_perCondition.{}'.format(fig_save_path, roi, save_format))


#%%           
#               ###########   2. stats    ###########

alpha = 0.05

#%% 
# ############ 2.1 
# T-Testing IBC between freq bands
# ############ 
for band in bands:
    conVal_freqSub_NS = ibc_df['NS']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))
    conVal_freqSub_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))

    print("\n"+band)# Print band then average value
    print("Mean IBC in NS: {}".format(ibc_df['NS']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean())) 
    print("Mean IBC in ES: {}".format(ibc_df['ES']["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean())) 
    
    print(scipy.stats.shapiro(conVal_freqSub_NS))
    print(scipy.stats.shapiro(conVal_freqSub_ES))

    print('normaltest teststat = {} pvalue = {}'.format(*scipy.stats.normaltest(conVal_freqSub_ES)))
    print('normaltest teststat = {} pvalue = {}'.format(*scipy.stats.normaltest(conVal_freqSub_NS)))

    # Conduct Paired Sample T-Test on IBC 
    tstatistic, pvalue= scipy.stats.ttest_rel(conVal_freqSub_NS, conVal_freqSub_ES)
    print('\tT-Test Pvalue = {}'.format(pvalue))
    if pvalue < 0.05:
        print("\tSignificant difference across condition")
    else:
        print("\tnon-significant")
#%% 
# ############ 2.2 
# T-Testing IBC between freq band x ROIs
# ############ 

for band in bands:
    for roi in ROIs.keys():

        cut = basicAnalysis_tools.get_ch_idx(roi=roi, n_ch=n_ch, quadrant='inter')

        conVal_freqSub_ES = ibc_df['ES']["ccorr"][fqb2idx[band]][:, cut[0], cut[1]].mean(axis=(1, 2))
        conVal_freqSub_NS = ibc_df['NS']["ccorr"][fqb2idx[band]][:, cut[0], cut[1]].mean(axis=(1, 2))

        print("\n{}".format(band))# Print band then average value
        print("-- {}".format(roi))# Print band then average value
        print("Mean IBC in NS: {}".format(ibc_df['NS']["ccorr"][fqb2idx[band]][:, cut[0], cut[1]].mean())) 
        print("Mean IBC in ES: {}".format(ibc_df['ES']["ccorr"][fqb2idx[band]][:, cut[0], cut[1]].mean())) 
        
        # print(scipy.stats.shapiro(conVal_freqSub_NS))
        # print(scipy.stats.shapiro(conVal_freqSub_ES))

        print('\t--- Test whether a sample differs from a normal distribution.\n\t(H0:sample comes from a normal distribution)')
        
        # First on ES
        tstatistic, pvalue_ES= scipy.stats.normaltest(conVal_freqSub_ES)
        if pvalue_ES < alpha:
            print("\tES - H0 rejected ({}) -> Distribution NON-NORMAL".format(np.round(pvalue_ES, 3)))
        else:
            print("\tES - H0 accepted ({}) -> Distribution NORMAL".format(np.round(pvalue_ES, 3)))

        # Now on NS
        tstatistic, pvalue_NS= scipy.stats.normaltest(conVal_freqSub_NS)
        if pvalue_NS < alpha:
            print("\tNS - H0 rejected ({}) -> Distribution NON-NORMAL".format(np.round(pvalue_NS, 3)))
        else:
            print("\tNS - H0 accepted ({}) -> Distribution NORMAL".format(np.round(pvalue_NS, 3)))

        print('\t---')

        # Test IBC across conditions
        # If both dataset are normaly distributed: Paired Sample T-Test
        if pvalue_NS < alpha and pvalue_ES < alpha: 
            tstatistic, pvalue= scipy.stats.ttest_rel(conVal_freqSub_NS, conVal_freqSub_ES)
            print('\tT-Test Pvalue = {}'.format(pvalue))
            if pvalue < 0.05:
                print("\t> SIGNIFICANT difference across condition")
            else:
                print("\t> Non-significant")    
        else: # Conduct non-parametric version of the paired T-test: The Wilcoxon signed-rank test
            tstatistic, pvalue = scipy.stats.wilcoxon(conVal_freqSub_NS, conVal_freqSub_ES)
            print('\tWilcoxon Pvalue = {}'.format(pvalue))
            if pvalue < 0.05:
                print("\n\t#############\n\t> SIGNIFICANT (difference across condition)\n\t#############")
            else:
                print("\t> Non-significant")
        


#%% TESTING
sample_size = ibc_df[condi]["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2)).shape[0]

meanIBC_condiFreq_long = np.empty((sample_size*len(bands)*len(dyads),3))

cnt=0
for condi in conditions:
    for band in bands:
        condi_lab = np.full((sample_size), condi)
        frequ_lab = np.full((sample_size), band)

        meanIBC_condiFreq_long[cnt] = ibc_df[condi]["ccorr"][fqb2idx[band]][:, 0:n_ch, n_ch:2*n_ch].mean(axis=(1, 2))
        cnt+=sample_size

#%%
tmmp = pd.DataFrame(meanIBC_condiFreq_long)
tmmp.columns = ['Condition', 'Frequencies_bands', 'IBC']
sns.boxplot(x="Frequencies_bands", y="IBC",
            hue="Condition", palette=["m", "g"],
            data=tmmp)

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