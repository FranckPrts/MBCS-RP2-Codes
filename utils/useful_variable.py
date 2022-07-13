from collections import OrderedDict

rejected_dyads = []

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

ch_frontal = [
    'Fp1', 'Fp2', 
    'F3', 'F4', 
    'F7', 'F8', 
    'Fz', 'AFz']

ch_temporal = ['T7', 'T8']

ch_occipital = ['O1', 'O2', 'POz']

ch_parietal = [
    'C3', 'C4', 
    'P3', 'P4', 
    'P7', 'P8', 
    'Cz', 'Pz', 
    'CPz']

assert len(ch_to_keep) == len(ch_frontal) + len(ch_parietal) + len(ch_occipital) + len(ch_temporal), 'The lengths of the channel list/sub-list does not match up.'

freq_bands = {'Theta': [4.0, 7.0],
              'Alpha-Low': [7.5, 11.0],
              'Alpha-High': [11.5, 13.0],
              'Beta': [13.5, 29.5],
              'Gamma': [30.0, 48.0]} 

freq_bands_ord = OrderedDict(freq_bands)

fqb2idx =  {'Theta': 0,
              'Alpha-Low': 1,
              'Alpha-High': 2,
              'Beta': 3,
              'Gamma': 4
              }