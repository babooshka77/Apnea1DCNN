from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter


def filter_final():
   'Read record values'
    record = loadmat("cleaned data/ECG2/a01_s1.mat") #read from a01_s1 matlab file
    val = record['val']
    val = np.array(val)
    
    'Scaler/MinMax Normalize only'
    scaler = MinMaxScaler()
    norm_val = scaler.fit_transform(val)
    time = [0]
    i=0
    x=0
    while i<len(val)-1:
        x = x + 0.01
        x  = round(x,4)
        time.append(x)
        i = i+1 
    cutoff=0.05
    sample_rate=100
    norm_val_tp = np.transpose(norm_val)
    data=norm_val_tp
    b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    filtered_data = filtfilt(b, a, data)
    return filtered_data, time