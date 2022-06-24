from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def normalization (record):
    global val
    'Read record values'
    record = loadmat("cleaned data/ECG2/a01_s1.mat") #read from a01_s1 matlab file
    val = record['val']
    val = np.array(val)

    'Scaler/MinMax Normalize only'
    scaler = MinMaxScaler()
    norm_val = scaler.fit_transform(val)
    
    return norm_val