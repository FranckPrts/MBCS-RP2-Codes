#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

# Data science
import json
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat


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

        for file in os.listdir(data_path+condi):       # Per condition
            if file.endswith(".set"):                  # Only look for SET files
                dyad = int(file.split('_')[4][0:-1]) /2      # find files' dyad 
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

reject_ch_manifest = dict()

def add_reject_ch_manifest(condition:str, dyad:str, sub_fname, reject:list, save_to:str = None):
    
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