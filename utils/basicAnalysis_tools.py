#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

# Data science
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat


def get_analysis_manifest(data_path:str, condition:list):
    
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
    
    return analysis_manifest