#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

# Data science
import glob
from importlib.util import spec_from_file_location
import json
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

reject_ch_manifest = dict()
reject_ch_manifest["total_reject_count"] = 0

def get_analysis_manifest(data_path:str, condition:list, save_to:str = None):

    '''
    A function to read the data/experimental condition and create a data manifest.
    
    Arguments:
        data_path: str
            Path to the folder containning the data. It is expected that the folder
            contains a folder for each experimental condition described in "condition"
            containning the participant's SET files.
        condition: list
            A list of the condition (folder names as str) to create the manifest for.
        save_to: str
            If provided, a path to the folder where the data_manifest will be saved.

    '''
    
    analysis_manifest = dict()
    
    for condi in condition : # loop through condition

        verbose_condition = condi.split('_')[1]
        analysis_manifest[verbose_condition]=dict()  # Instantiate a key for condition

        for file in os.listdir(data_path+condi):        # Per condition
            if file.endswith(".set"):                   # Only look for SET files
                dyad = int(file.split('_')[4][0:-1]) /2 # find files' dyad 
                dyad = str(dyad)[:-2]

                if dyad not in  analysis_manifest[verbose_condition]:
                    analysis_manifest[verbose_condition][dyad] = []
                    analysis_manifest[verbose_condition][dyad].append(file)
                    
                else:
                    analysis_manifest[verbose_condition][dyad].append(file)

    # info print
    print("Available condition", analysis_manifest.keys())
    for i in analysis_manifest:
        print("In condition {}, the {} availble dyads are: ".format(i, len(analysis_manifest[i].keys())))
        print(analysis_manifest[i].keys())
    
    print("\nThe dyads that are present in on conditions only: ")
    print(list(set(analysis_manifest[condition[1].split('_')[1]]).difference(analysis_manifest[condition[0].split('_')[1]])))
    
    if save_to is not None:

        json_string = json.dumps(analysis_manifest)
        f = open(save_to+"df_manifest.json","w")
        f.write(json_string)
        f.close()

    return analysis_manifest

def add_reject_ch_manifest(condition:str, dyad:str, sub_fname, reject:list, save_to:str = None):
    reject_ch_manifest["total_reject_count"] += 1 

    if condition not in reject_ch_manifest.keys():
        reject_ch_manifest[condition] = dict()
    if dyad not in reject_ch_manifest[condition].keys():
        reject_ch_manifest[condition][dyad] = dict()
    
    reject_ch_manifest[condition][dyad] = (sub_fname, reject)

    json_string = json.dumps(reject_ch_manifest) 
    f = open(save_to+"reject_ch_manifest.json","w")
    f.write(json_string)
    f.close()

    print(">> Reject of {} for {} (dyad {}) in condition {}".format(reject, sub_fname, dyad, condition))

def create_ibc_manifest(data_path:str, mani_path:str, conditions:list, ibc_metrics:list, n_ch:int, nb_freq_band:int, save:bool = True, specific_file:str = None):
    
    # If the user just want to load a file. Caution! the output is then different.
    if specific_file is not None:
        result = np.load("{}{}".format(data_path, specific_file))
        return result

    f = open(mani_path+"reject_ch_manifest.json","r")
    reject_ch_manifest = json.load(f) 
    f.close()

    ibc_manifest = dict()
    
    file_nb        = len(glob.glob(data_path+"*.npy"))
    dyad_per_condi = int(file_nb//len(conditions))
    dyad_per_condi -= reject_ch_manifest["total_reject_count"]//2

    for condi in conditions : 
        ibc_manifest[condi]=dict() 
        for metric in ibc_metrics:
            ibc_manifest[condi][metric] = np.zeros([dyad_per_condi, nb_freq_band, n_ch*2, n_ch*2], dtype=np.float32)

    cnt_es = 0
    cnt_ns = 0
    reject = []

    for file in glob.glob(data_path+"*.npy"):
        #  Assuming the following file convention: dyad_{DYAD#}_condition_{CONDITION}_IBC_{ccorr, plv...}.npy
        _, dyad, _, condi, _, ibc_metric = file.split("/")[-1].split('_')
        ibc_metric = ibc_metric[:-4]
        result = np.load(file)

        if result.shape != (nb_freq_band, n_ch*2, n_ch*2):
            print("Not dealing with dyad #{} (cond: {}) because its 'result' has shape {} instead of {}".format(dyad, condi, result.shape, (nb_freq_band, n_ch*2, n_ch*2)))
            reject.append([dyad, condi])
            pass
        else:
            if condi == 'ES':
                ibc_manifest[condi][ibc_metric][cnt_es] = result
                cnt_es += 1
            if condi == 'NS':
                ibc_manifest[condi][ibc_metric][cnt_ns] = result
                cnt_ns += 1

    # Reshaping the data so it can be unpacked in its freq dimension easily
    for condi in conditions:
        for ibc in ibc_metrics:
            ibc_manifest[condi][ibc] = np.moveaxis(ibc_manifest[condi][ibc], 1, 0)

    if save:
        json_string = json.dumps(ibc_manifest, cls=NpEncoder)
        f = open(mani_path+"ibc_manifest.json","w")
        f.write(json_string) 
        f.close()

    return ibc_manifest, reject

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_ch_idx(roi:str(), n_ch:int(), quadrant:str()):

    from .useful_variable import ROIs, ch2idx

    assert quadrant in ['inter', 'intra_A', 'intra_B'], "Quadrand is wrong"
    assert roi in ['Frontal', 'Temporal', 'Occipital', 'Parietal'], "Roi is not defined"
    
    cut1 = []
    for ch in ROIs[roi]:
        cut1.append(ch2idx[ch])
    cut2 = cut1

    if quadrant == 'inter':
        cut2 = [x+n_ch for x in cut2]
    elif quadrant == 'intra_A':
        pass # The idx already indicate these locs
    elif quadrant == 'intra_B':
        cut1 = [x+n_ch for x in cut1]
        cut2 = [x+n_ch for x in cut2]
    
    cut = np.ix_(cut1, cut2)

    return cut

