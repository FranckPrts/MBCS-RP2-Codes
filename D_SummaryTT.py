#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#           ###########   Imports and loading data    ###########

#%% ---------------------------------------------------------------<>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon, ttest_rel, pearsonr
import seaborn as sns



tt_path        = '../SNS_Data_Fall_2020/video-files/TurnTaking_allCondi_allDyads.csv'
ibc_path       = '../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/ibc_perDyad_perCondi.csv'
save_plot_path = '../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/plots/TurnTaking/'

tt_df  = pd.read_csv(tt_path, index_col=False)
ibc_df = pd.read_csv(ibc_path, index_col=False)

#%%
#           ###########   Summarize data    ###########
tt_es=tt_df[tt_df['Condition']=='ES']
tt_ns=tt_df[tt_df['Condition']=='NS']

print('Condition es\n', tt_es.describe())
print('\nCondition NS\n', tt_ns.describe())

# tt_df.groupby('Condition')['SuccTT'].agg([np.mean, np.std, np.min, np.max])
# tt_df.groupby('Condition')['UnsuccTT'].agg([np.mean, np.std, np.min, np.max])






#           ###########   Does the amount of succTT differs across conditions?       ###########

#%%
# Ploting density for NS & Testing for normality in ES
sns.kdeplot(tt_ns.SuccTT).set(title='Turntaking count density (NS)\nn=35')
plt.savefig(save_plot_path + 'succTT_density_NS.pdf')
print('Shapiro (NS): teststat = {} pvalue = {}'.format(*shapiro(tt_ns.SuccTT)))
# Shapiro (NS): teststat = 0.9818142652511597 pvalue = 0.8165109753608704

#%%
# Ploting density for ES & Testing for normality in ES
sns.kdeplot(tt_es.SuccTT).set(title='Turntaking count density (ES)\nn=35')
plt.savefig(save_plot_path + 'succTT_density_ES.pdf')
print('Shapiro (ES): teststat = {} pvalue = {}'.format(*shapiro(tt_es.SuccTT)))
# Shapiro (ES): teststat = 0.9087592363357544 pvalue = 0.006858120672404766

#%%
# For alpha = 0.05, we see that the ES TT distribution is not normal. We will thus conduct
# the non-parametric version of the paired T-test: The Wilcoxon signed-rank test.

tstatistic, pvalue = wilcoxon(tt_es.SuccTT, tt_ns.SuccTT)
print('\tWilcoxon: p={}, t={}'.format(np.round(pvalue, 3), np.round(tstatistic, 3)))
print("\n\t###\n\t>>> SIGNIFICANT (difference across condition)\n\t###") if pvalue < 0.05 else print("\t>>> Non-significant <<<")

	# Wilcoxon: p=0.0, t=69.5

	# ###
	# >>> SIGNIFICANT (difference across condition)
	# ###

#           ###########   Does  the amount of UnsuccTT differs across conditions?       ###########

#%%
# Ploting density for NS & Testing for normality in NS
sns.kdeplot(tt_ns.UnsuccTT).set(title='Unsuccesful Turntaking count density (NS)\nn=35')
plt.savefig(save_plot_path + 'UnsuccTT_density_NS.pdf')
print('Shapiro (NS): teststat = {} pvalue = {}'.format(*shapiro(tt_ns.UnsuccTT)))
# Shapiro (NS): teststat = 0.6848371624946594 pvalue = 2.1840246233750804e-07

#%%
# Ploting density for ES & Testing for normality in ES
sns.kdeplot(tt_es.UnsuccTT).set(title='Unsuccesful Turntaking count density (ES)\nn=35')
plt.savefig(save_plot_path + 'UnsuccTT_density_ES.pdf')
print('Shapiro (ES): teststat = {} pvalue = {}'.format(*shapiro(tt_es.UnsuccTT)))
# Shapiro (ES): teststat = 0.6067503690719604 pvalue = 1.7514395267426153e-08

#%%
# Both distribution are normal, we're using a Paired sample t-test

tstatistic, pvalue = ttest_rel(tt_es.UnsuccTT, tt_ns.UnsuccTT)
print('Paired Student T-Test: p={}, t={}'.format(np.round(pvalue, 3), np.round(tstatistic, 3)))
print("\n\t###\n\t>>> SIGNIFICANT (difference across condition)\n\t###") if pvalue < 0.05 else print("\t>>> Non-significant <<<")

    # Paired Student T-Test: p=0.571, t=-0.572
	# >>> Non-significant <<<


#           ###########   Does the amount of interjections differs across conditions?       ###########

#%%
# Ploting density for NS & Testing for normality in NS
sns.kdeplot(tt_ns.Interjection).set(title='Interjection count density (NS)\nn=35')
plt.savefig(save_plot_path + 'Interjection_density_NS.pdf')
print('Shapiro (NS): teststat = {} pvalue = {}'.format(*shapiro(tt_ns.Interjection)))
# Shapiro (NS): teststat = 0.9731582999229431 pvalue = 0.5356448888778687

#%%
# Ploting density for ES & Testing for normality in ES
sns.kdeplot(tt_es.Interjection).set(title='Interjection count density (ES)\nn=35')
plt.savefig(save_plot_path + 'Interjection_density_ES.pdf')
print('Shapiro (ES): teststat = {} pvalue = {}'.format(*shapiro(tt_es.Interjection)))
# Shapiro (ES): teststat = 0.9790245294570923 pvalue = 0.7271564602851868

#%%
# Both distribution are normal, we're using a Paired sample t-test

tstatistic, pvalue = wilcoxon(tt_es.Interjection, tt_ns.Interjection)
print('Paired Student T-Test: p={}, t={}'.format(np.round(pvalue, 3), np.round(tstatistic, 3)))
print("\n\t###\n\t>>> SIGNIFICANT (difference across condition)\n\t###") if pvalue < 0.05 else print("\t>>> Non-significant <<<")

	# Paired Student T-Test: p=0.017, t=169.5

	# ###
	# >>> SIGNIFICANT (difference across condition)
	# ###



# %%

es_ibc_df = ibc_df.loc[:,ibc_df.columns.str.contains('ES')]
ns_ibc_df = ibc_df.loc[:,ibc_df.columns.str.contains('NS')]

# %%

merged_es = pd.merge(left=tt_es, right=es_ibc_df, left_on='Dyad', right_on='Dyad_ES')
merged_ns = pd.merge(left=tt_ns, right=ns_ibc_df, left_on='Dyad', right_on='Dyad_NS')

# %%
# Compute correlation between count of successful 
# turntaking instance and IBC in 2 criteria (spatial, frequency band)

es_r, es_p = pearsonr(merged_es['SuccTT'], merged_es['ES_IBC'])
print('Pearson moment-product correlation (ES): r={}, p={}'.format(np.round(es_r, 3), np.round(es_p, 3)))
ns_r, ns_p = pearsonr(merged_ns['SuccTT'], merged_ns['NS_IBC'])
print('Pearson moment-product correlation (NS): r={}, p={}'.format(np.round(ns_r, 3), np.round(ns_p, 3)))

    # Pearson moment-product correlation (ES): r=0.174, p=0.35
    # Pearson moment-product correlation (NS): r=0.13, p=0.486
# %%
# Compute correlation between count of UNsuccessful 
# turntaking instance and IBC in 2 criteria (spatial, frequency band)

es_r, es_p = pearsonr(merged_es['UnsuccTT'], merged_es['ES_IBC'])
print('Pearson moment-product correlation (ES): r={}, p={}'.format(np.round(es_r, 3), np.round(es_p, 3)))
ns_r, ns_p = pearsonr(merged_ns['UnsuccTT'], merged_ns['NS_IBC'])
print('Pearson moment-product correlation (NS): r={}, p={}'.format(np.round(ns_r, 3), np.round(ns_p, 3)))

    # Pearson moment-product correlation (ES): r=0.076, p=0.685
	# Pearson moment-product correlation (NS): r=-0.07, p=0.709
# %%
# Compute correlation between count of Interjection
#  instance and IBC in 2 criteria (spatial, frequency band)

es_r, es_p = pearsonr(merged_es['Interjection'], merged_es['ES_IBC'])
print('Pearson moment-product correlation (ES): r={}, p={}'.format(np.round(es_r, 3), np.round(es_p, 3)))
ns_r, ns_p = pearsonr(merged_ns['Interjection'], merged_ns['NS_IBC'])
print('Pearson moment-product correlation (NS): r={}, p={}'.format(np.round(ns_r, 3), np.round(ns_p, 3)))

    # Pearson moment-product correlation (ES): r=-0.076, p=0.686
	# Pearson moment-product correlation (NS): r=0.183, p=0.326

#%%
stacked_IBC_TT = merged_es
stacked_IBC_TT.rename(columns={'ES_IBC':'IBC'}, inplace=True)

tmp = merged_ns
tmp.rename(columns={'NS_IBC':'IBC'}, inplace=True)
 
stacked_IBC_TT = pd.concat([stacked_IBC_TT, tmp], axis=0, ignore_index=True)

del tmp

#%%
sns.relplot(data=stacked_IBC_TT, x="SuccTT", y="IBC", hue="Condition").set(title='Count of successful turn-taking against IBC\nn=31')
plt.savefig(save_plot_path + 'SuccTT-vs-IBC_bothCondi.pdf')

#%%
sns.relplot(data=stacked_IBC_TT, x="UnsuccTT", y="IBC", hue="Condition").set(title='Count of successful turn-taking against IBC\nn=31')
plt.savefig(save_plot_path + 'UnsuccTT-vs-IBC_bothCondi.pdf')

#%%
sns.relplot(data=stacked_IBC_TT, x="Interjection", y="IBC", hue="Condition").set(title='Count of successful turn-taking against IBC\nn=31')
plt.savefig(save_plot_path + 'Interjection-vs-IBC_bothCondi.pdf')


#%%
# sns.scatterplot(merged_es['SuccTT'], merged_es['ES_IBC']).set(title='IBC ')
# sns.scatterplot(merged_ns['SuccTT'], merged_ns['NS_IBC']).set(title='IBC ')

# sns.regplot(x="SuccTT", y="ES_IBC", data=merged_es) # Not using this since there is no correlation


# ax = sns.barplot(x="day", y="total_bill", hue="sex", data=stacked_IBC_TT, palette="winter_r")
