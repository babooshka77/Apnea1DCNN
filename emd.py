from PyEMD import EMD, Visualisation
import numpy as np
import pandas as pd

def emd_signal(filtered_data_tp,time):
   'Define time array from time in filter_final funct.'
    time_arr = np.array(time) 
    t = time_arr 
    'S_tp = Signal Input from Filtered+Normalized Signal'
    S_tp = filtered_data_tp.flatten()

    'EMD Process'
    emd = EMD() #run EMD process
    emd.emd(S_tp)
    imfs, res = emd.get_imfs_and_residue()
    data = pd.DataFrame(data = imfs) #data = data IMF hasil EMD

    
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
    'from 1 x 360.000 reshape to 60x6000, 60 rows and 6000 cols'
    data_i_1_conv = data_i_1_conv.reshape(60,6000)
    data_i_2_conv = data_i_1_conv.reshape(60,6000)
    data_i_3_conv = data_i_1_conv.reshape(60,6000)
    data_i_4_conv = data_i_1_conv.reshape(60,6000)

    
    'conv back to pandas dataframe in order to conv them to csv files...'
    data_i_1_conv = pd.DataFrame(data_i_1_conv)
    data_i_2_conv = pd.DataFrame(data_i_2_conv)
    data_i_3_conv = pd.DataFrame(data_i_3_conv)
    data_i_4_conv = pd.DataFrame(data_i_4_conv)

    return data_i_1_conv, data_i_2_conv, data_i_3_conv, data_i_4_conv
