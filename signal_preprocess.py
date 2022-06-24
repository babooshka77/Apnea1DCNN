# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:21:09 2021

@author: fnx
"""
import matplotlib.pyplot as plt
import scipy 
import wfdb
import numpy as np
import pandas as pd
import os
import sklearn
'Filter'
from scipy import signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter

'iterasi'
import os
import pathlib

'Scaler'
from sklearn.preprocessing import MinMaxScaler

'EMD'
from PyEMD import EMD, Visualisation
import numpy as np
os.chdir('E:/Felix/1822041/Tugas Akhir/')

'Timer'
from timeit import default_timer as timer
from datetime import timedelta

'Iterasi'
letter = ['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','b01','b02','b03','b04','b05','c01','c02','c03','c04','c06','c07','c08','c09','c10']
num = [8,8,8,8,7,8,8,8,8,8,7,8,8,8,8,8,6,8,8,8,8,8,7,7,7,8,8,7,8,7,7,8,7,7]


data_i_1_conv_temp =  pd.DataFrame([])
data_i_2_conv_temp =  pd.DataFrame([])
data_i_3_conv_temp =  pd.DataFrame([])
data_i_4_conv_temp =  pd.DataFrame([])
start = timer()

'all imfs'
def main():
    
    i = 0
    j = 1

    while i < len(letter):
        while j <= num[i]:
            record_num = letter[i] + '_s' + str(j) 
            record_dir = "cleaned data/ECG2/" + record_num + '.mat'

            record = loadmat(record_dir)
            
            filtered_data, time,norm_val = filter_final(record)
            filtered_data_tp = np.transpose(filtered_data)
            data_i_1, data_i_2, data_i_3, data_i_4, data_i_1_conv, data_i_2_conv, data_i_3_conv, data_i_4_conv = emd_signal(filtered_data_tp,time)
            save_file_imf_1(data_i_1, data_i_1_conv, record_num,time)
            save_file_imf_2(data_i_2,data_i_2_conv,record_num,time)
            save_file_imf_3(data_i_3,data_i_3_conv,record_num,time)
            save_file_imf_4(data_i_4,data_i_4_conv,record_num,time)
            
            j+=1
        j=1
        i+=1
      
def filter_final (record):
    global val, filtered_data, time, norm_val
    'File input'

    val = record['val']
    val = np.array(val)
    val = np.transpose(val)
    '1 jam'
    
    val = val
 
    
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
    'Filtering with Notch Filter'
    b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    filtered_data = filtfilt(b, a, data)
    filtered_data_tp = np.transpose(filtered_data)
    return filtered_data, time, norm_val




'EMD Signal Function'
def emd_signal(filtered_data_tp,time):
    global data
    global data_i_1
    global data_i_1_conv, data_i_2_conv, data_i_3_conv, data_i_4_conv
    global data_i_2
    global data_i_3
    global data_i_4
    global imfs, data_i_1_conv_reshape , data_i_2_conv_reshape, data_i_3_conv_reshape, data_i_4_conv_reshape
    # global data_i_1
    time_arr = np.array(time) 
    t = time_arr 
    S_tp = filtered_data_tp.flatten()
    emd = EMD()
    emd.emd(S_tp)
    imfs, res = emd.get_imfs_and_residue()
    data = pd.DataFrame(data = imfs)
    
    'ambil imf ke-1,2,3,4 dari row ke 0,1,2,3'
    data_i_1 = data.iloc[[0]]
    data_i_2 = data.iloc[[1]]
    data_i_3 = data.iloc[[2]]
    data_i_4 = data.iloc[[3]]
    
    'convert dataframes to np in order to use np.reshape()'
    data_i_1_conv = data_i_1.to_numpy()
    data_i_2_conv = data_i_2.to_numpy()
    data_i_3_conv = data_i_3.to_numpy()
    data_i_4_conv = data_i_4.to_numpy()
    'from 1 x 360.000 reshape to 60x6000, 60 rows + 6000 cols'
    data_i_1_conv = data_i_1_conv.reshape(60,6000)
    print('IMF 1 data has been reshaped to :', data_i_1_conv.shape)
    data_i_2_conv = data_i_2_conv.reshape(60,6000)
    print('IMF 2 data has been reshaped to :', data_i_2_conv.shape)
    data_i_3_conv = data_i_3_conv.reshape(60,6000)
    print('IMF 3 data has been reshaped to :', data_i_3_conv.shape)
    data_i_4_conv = data_i_4_conv.reshape(60,6000)
    print('IMF 4 data has been reshaped to :', data_i_4_conv.shape)
    
    data_i_1_conv_reshape = data_i_1_conv.transpose()
    data_i_2_conv_reshape = data_i_2_conv.transpose()
    data_i_3_conv_reshape = data_i_3_conv.transpose()
    data_i_4_conv_reshape = data_i_4_conv.transpose()
    
    'conv back to pandas dataframe in order to conv them to csv files...'
    data_i_1_conv = pd.DataFrame(data_i_1_conv)
    print('IMF 1 data has been converted to DataFrame with shape : ', data_i_1_conv.shape)
    data_i_2_conv = pd.DataFrame(data_i_2_conv)
    print('IMF 2 data has been converted to DataFrame with shape : ', data_i_2_conv.shape)
    data_i_3_conv = pd.DataFrame(data_i_3_conv)
    print('IMF 3 data has been converted to DataFrame with shape : ', data_i_3_conv.shape)
    data_i_4_conv = pd.DataFrame(data_i_4_conv)
    print('IMF 4 data has been converted to DataFrame with shape : ', data_i_4_conv.shape)

    return data_i_1, data_i_2, data_i_3, data_i_4, data_i_1_conv, data_i_2_conv, data_i_3_conv, data_i_4_conv

    

'save imf data'
def save_file_imf_1(data_i_1, data_i_1_conv, record_num,time):
    global data_i_1_conv_temp
    data_i_1_conv_temp = data_i_1_conv_temp.append(data_i_1_conv, ignore_index=True)
    data_i_1_conv_temp.to_csv('E:/Felix/1822041/Tugas Akhir/new/imf_1/record_imf1_' + record_num + '.csv',index=False)
    print('Success! CSV Format of IMF1 Record from ',record_num,' with shape : ',data_i_1_conv_temp.shape)

    
def save_file_imf_2(data_i_2,data_i_2_conv,record_num,time):
    global data_i_2_conv_temp
    data_i_2_conv_temp = data_i_2_conv_temp.append(data_i_2_conv, ignore_index=True)
    data_i_2_conv_temp.to_csv('E:/Felix/1822041/Tugas Akhir/new/imf_2/record_imf2_' + record_num +  '.csv',index=False)
    print('Success! CSV Format of IMF2 Record from ',record_num,' with shape : ',data_i_2_conv_temp.shape)
    
def save_file_imf_3(data_i_3,data_i_3_conv, record_num,time):
    global data_i_3_conv_temp
    data_i_3_conv_temp = data_i_3_conv_temp.append(data_i_3_conv, ignore_index=True)
    data_i_3_conv_temp.to_csv('E:/Felix/1822041/Tugas Akhir/new/imf_3/record_imf3_'  + record_num + '.csv',index=False)
    print('Success! CSV Format of IMF3 Record from ',record_num,' with shape : ',data_i_3_conv_temp.shape)
    
def save_file_imf_4(data_i_4,data_i_4_conv, record_num,time):
    global data_i_4_conv_temp
    data_i_4_conv_temp = data_i_4_conv_temp.append(data_i_4_conv, ignore_index=True)
    data_i_4_conv_temp.to_csv('E:/Felix/1822041/Tugas Akhir/new/imf_4/record_imf4_'  + record_num + '.csv',index=False)
    print('Success! CSV Format of IMF4 Record from ',record_num,' with shape : ',data_i_4_conv_temp.shape)
    
'run program'
if __name__ == "__main__":
    main()    
    end = timer()
    print('Time elapsed: ', timedelta(seconds=end-start))
    
    