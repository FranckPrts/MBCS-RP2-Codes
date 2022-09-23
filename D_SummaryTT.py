#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

#%%
from heapq import merge
import numpy as np
import pandas as pd

tt_path = '../SNS_Data_Fall_2020/video-files/TurnTaking_allCondi_allDyads.csv'
ibc_path ='../SNS_Data_Fall_2020/EEG/Cleaned_EEG/MBCS-RP2-Results/ibc_perDyad_perCondi.csv'

tt_df  = pd.read_csv(tt_path, index_col=False)
ibc_df = pd.read_csv(ibc_path, index_col=False)

#%%
tt_df.groupby('Condition')['SuccTT'].agg([np.mean, np.std, np.min, np.max])

#%%
tt_es=tt_df[tt_df['Condition']=='ES']
tt_ns=tt_df[tt_df['Condition']=='NS']

print('es\n', tt_es.describe())
print('ns\n', tt_ns.describe())

# %%

es_ibc_df = ibc_df.loc[:,ibc_df.columns.str.contains('ES')]
ns_ibc_df = ibc_df.loc[:,ibc_df.columns.str.contains('NS')]

# %%

merged_es = pd.merge(left=tt_es, right=es_ibc_df, left_on='Dyad', right_on='Dyad_ES')
merged_ns = pd.merge(left=tt_ns, right=ns_ibc_df, left_on='Dyad', right_on='Dyad_NS')

# %%
# Compute correlation between count of successful 
# turntaking instance and IBC in 2 criteria (spatial, frequency band)

print(merged_es['SuccTT'].corr(merged_es['ES_IBC']))
print(merged_ns['SuccTT'].corr(merged_ns['NS_IBC']))
# %%
