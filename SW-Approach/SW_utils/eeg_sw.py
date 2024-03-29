#!/usr/bin/env python
# coding=utf-8
# @author Franck Porteous <franck.porteous@proton.me>

'''
Tools to import an EEG SET file (EEGLAB) into MNE without losing on the 
metadata that would be stored in the MataLab EEG.event struct.

The class also read SET files and enalbe selecting data falling in a sliding window.
'''

import mne
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

    def __init__(self, path:str, sw_min:float, sw_max:float, ch_to_keep:list):
        
        self.path = path
        self.sw_min = sw_min
        self.sw_max = sw_max
        
        self.df        = self.read_set_events(self.path)
        self.df_sw     = pd.DataFrame
        self.df_sw_mne = np.ndarray
        
        if sw_min and sw_max is not None:
            self.event_slider(self.sw_min, self.sw_max)

        self.ch_to_keep = ch_to_keep

        # reading the epoch file
        self.eeg = mne.io.read_epochs_eeglab(self.path).pick_channels(self.ch_to_keep, ordered=False)
        self.sfreq = self.eeg.info['sfreq']
        # selecting channels of interest
        # self.eeg = self.eeg.copy()

    
    def read_set_events(self, path, ignore_fields=None):
        
        """
        Open set file, read EEG.event struct and turn it into a pandas.df.

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
                    struct_as_record=False, squeeze_me=True, appendmat=False)['EEG']
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
        self.sw_min, self.sw_max = sw_min, sw_max  # Re-assigning sw_min, sw_max in case the first class instentiation didn't
        assert(sw_min < sw_max), "Problem: sw_min is greater than sw_max" 
        self.df_sw = self.df.loc[(self.df['epoch_id'] >= sw_min) & (self.df['epoch_id'] <= sw_max)]

    def custom_event_to_MNE(self):
        events = self.df_sw

        # Add the needed 'dontuse' column to `event_csv` dataframe before formating it for MNE
        # This column can also serve as a filter for bad events
        
        # print(type(events))
        # events_selected = pd.DataFrame
        events_selected = events.loc[:,['latency','epoch']]                 # select columns required by MNE
        events_selected['dontuse'] = 0                                      # Add necessary 'dontuse' column
        events_selected = events_selected[['latency', 'dontuse', 'epoch']]  # Reorder the columns

        self.event_id_sw = list()
        for i in events_selected['epoch'].unique():
            self.event_id_sw = str(i) 

        self.df_sw_mne = events_selected.to_numpy()

    def create_custom_metadata(self):
    

        meta_tmin, meta_tmax = 0.0, 1.0 # In seconds

        # Define our ecode of interrest 
        # ecodeOI = ['10022', '501024']

        self.metadata_sw, events_sw, event_id = mne.epochs.make_metadata(
            
            events=self.df_sw_mne,
            event_id=self.event_id_sw,
            tmin= meta_tmin, tmax= meta_tmax,
            sfreq=self.sfreq
            # row_events=ecodeOI, # select only event of choices
            )

        events = events.astype(int)

        self.metadata_sw.head()
    
    def convert_EEG_to_MNE(self):

        print("Converting EEG data to mne.Raw ..."
        "\n\tUsing effective time window [{} - {}]"
        "\n\tHere, from epoch #{} to #{}.".format(
            self.sw_min, self.sw_max,
            self.df_sw["epoch"].iat[0], self.df_sw["epoch"].iat[-1]))
        
        self.custom_event_to_MNE()
        # self.create_custom_metadata()    

        self.mne_epo_sw = self.eeg[self.event_id_sw]  
        print('Done', self.mne_epo_sw)
        
