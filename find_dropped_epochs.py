#%%
from utils import eeg_sw
import os

path = "../SNS_Data_Fall_2020/EEG/Cleaned_EEG/EEG_data_cleaned/SNS_ES_cleaned/"
# The path must be changed for each folder containning SET files

l_files = os.listdir(path)

dd = dict()
#%%
for i in l_files:
    if i[0] == "." or i[-3:] != "set":
        pass
    else:
        print(path+i)
        sub = eeg_sw.import_set_custom(path+i, sw_min=None, sw_max=None)
        dd[i] = sub.df["epoch_id"].to_numpy()

#  The dict "dd" now contains for each key (filename with the appendix .set) 
#  a value being the numpy array with the epochs that weren't dropped.
# %%
len(dd)

# To access the retainned epochs of a subject, you may use:
dd["THE-FILE-NAME.set"]
