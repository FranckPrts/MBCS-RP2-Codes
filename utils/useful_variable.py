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

ROIs = {'Frontal':   ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'AFz'],
        'Temporal':  ['T7', 'T8'],
        'Occipital': ['O1', 'O2', 'POz'],
        'Parietal':  ['C3', 'C4', 'P3', 'P4', 'P7', 'P8', 'Cz', 'Pz', 'CPz'],
        'Selected_sensors':  ['Fz', 'Cz']
        }

freq_bands = {'Theta': [4.0, 7.0],
              'Alpha-Low': [7.5, 11.0],
              'Alpha-High': [11.5, 13.0],
              'Beta': [13.5, 29.5],
              'Gamma': [30.0, 48.0],
              'Selected_freqBand': [18.0, 22.0]} 

freq_bands_ord = OrderedDict(freq_bands)

fqb2idx =  {'Theta': 0,
              'Alpha-Low': 1,
              'Alpha-High': 2,
              'Beta': 3,
              'Gamma': 4,
              'Selected_freqBand':0
              }


ch2idx = {}
for idx, ch in enumerate(ch_to_keep):
    ch2idx[ch]=idx