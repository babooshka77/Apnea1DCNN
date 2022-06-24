import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_2 = pd.read_csv('E:/Felix/1822041/Tugas Akhir/new/imf_1/recordAnnot_imf1_final.csv')
df_2 = df_2.to_numpy()
annot = pd.read_csv('E:/Felix/1822041/Tugas Akhir/annot/annot_binary_final.csv')
'Scale first then train-test-split'
print('Begin Scaler')
scaler = StandardScaler()
df_2_scaled = scaler.fit_transform(df_2)
print('df scaled with StandardScaler()')


'Expand dims before train-test-split'
df_2_scaled = np.expand_dims(df_2_scaled, axis=-1)
def train(df_2_scaled, annot_map):
    print('Begin train funct')
    global X_train, y_train
    global X_test, y_test
    global X_val, y_val
    global train_x_scaled,test_x_scaled 
    
    X_train, X_test, y_train, y_test = train_test_split(df_2_scaled, annot_map, test_size=0.2,stratify=np.array(annot))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,stratify=np.array(y_train))
    'Tipe data float32'
    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    print('X_train shape: ', X_train.shape,'y_train shape: ', y_train.shape)
    print('X_test shape: ', X_test.shape,'y_test shape: ', y_test.shape)
    print('X_val shape: ', X_val.shape,'y_val shape: ', y_val.shape)
    return X_train, y_train, X_test, y_test, X_val, y_val