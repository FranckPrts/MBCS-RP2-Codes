#!/usr/bin/env python
# coding=utf-8

'''
Tools to import an EEG SET file (EEGLAB) into MNE without losing on the 
metadata that would be stored in the MataLab EEG.event struct.

'''

import numpy as np
import pandas as pd

from scipy.io import loadmat

class import_set_custom ():
    
    """
    A class to read the (all) metadata stored in SET 
    files. 

    Arguments:
        path: path
            Path to SET file (ablsolute or relative)
        sw_min: float
            Minimum boudary for the sliding window 
        sw_mxn: float
            Maximum boudary for the sliding window
    """

    def __init__(self, path:str, sw_min:float, sw_max:float):
        
        self.path = path
        self.sw_min = sw_min
        self.sw_max = sw_max
        
        self.df        = self.read_set_events(self.path)
        self.df_sw     = pd.DataFrame
        self.df_sw_mne = np.ndarray
        
        if sw_min and sw_max is not None:
            self.event_slider(self.sw_min, self.sw_max)

    
    def read_set_events(self, path, ignore_fields=None):
        
        """
        Open set file, read events and turn them into a dataframe

        Arguments:
            filename: str
                Name of the set file to read (absolute or relative path)
            ignore_fields: list of str | None
                Event fields to ignore
        
        Returns:
            df: pandas.DatFrame
                Events read into a (panda) dataframe

        Note:
            - This function is sourced from https://github.com/mne-tools/mne-python/issues/3837#issuecomment-266460434
        """

        EEG = loadmat(path, uint16_codec='latin1',
                    struct_as_record=False, squeeze_me=True)['EEG']
        flds = [f for f in dir(EEG.event[0]) if not f.startswith('_')]
        events = EEG.event
        df_dict = dict()
        for f in flds:
            df_dict[f] = [ev.__getattribute__(f) for ev in events]
        df = pd.DataFrame(df_dict)

        # reorder columns:
        take_fields = ['epoch', 'type']
        ignore_fields = list() if ignore_fields is None else ignore_fields
        take_fields.extend([col for col in df.columns if not
                        (col in take_fields or col in ignore_fields)])
        
        return df.loc[:, take_fields]
    
    def event_slider(self, sw_min:float, sw_max:float):    
        assert(sw_min < sw_max), "sw_min < sw_max"
        self.df_sw = self.df.loc[(self.df['epoch_id'] >= sw_min) & (self.df['epoch_id'] <= sw_max)]
    
    def convert_to__MNE_event(self):

        events = self.df_sw

        # Add the needed 'dontuse' column to `event_csv` dataframe before formating it for MNE
        # This column can also serve as a filter for bad events
        
        # print(type(events))
        # events_selected = pd.DataFrame
        events_selected = events.loc[:,['latency','epoch']]                 # select columns required by MNE
        events_selected['dontuse'] = 0                                      # Add necessary 'dontuse' column
        events_selected = events_selected[['latency', 'dontuse', 'epoch']]  # Reorder the columns

        self.df_sw_mne = events_selected.to_numpy()

        # return events_selected