# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 02:02:13 2022

@author: fnx
"""
   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from datetime import timedelta
import seaborn as sns
'Standard Scaler'
from sklearn.preprocessing import StandardScaler

'Deep Learning'
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import os
os.chdir('E:/Felix/1822041/Tugas Akhir/')

'Confusion matrix'
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics

'Specificity+Sensitivity'
import imblearn

def main():
    global df_final,df_final_scaled
    global annot, annot_map
    global imf_no 
    global test_no, conv_no, dense_no, filt_no
    imf_no = '2'
    test_no = '99'
    conv_no = '10'
    dense_no = '6'
    filt_no ='60'
    print('Begin main funct')
    
    df_final_dir = 'E:/Felix/1822041/Tugas Akhir/new/imf_'+imf_no+'/0imf_'+imf_no+'_sorted.csv'
    print('Now reading... : '+df_final_dir)
    df_final = pd.read_csv(df_final_dir)
    print('Success reading record file from: ', df_final_dir)
    'np array to expand dim'
    df_final = df_final.to_numpy()
    print('Success converting ', df_final_dir, ' to np format')
    
    'Scale first then train-test-split'
    print('Begin Scaler')
    scaler = StandardScaler()
    df_final_scaled = scaler.fit_transform(df_final)
    print('df scaled with StandardScaler()')
    
    
    'Expand dims before train-test-split'
    df_final_scaled = np.expand_dims(df_final, axis=-1)
    
    'Read annotation files'
    annot_dir = 'E:/Felix/1822041/Tugas Akhir/new/imf_'+imf_no+'/annot_sorted_imf'+imf_no+'.csv'
    annot = pd.read_csv(annot_dir)
    print('Annotation shape: ', annot.shape)
   
    # annot = pd.read_csv('D:/Kuliyah/.Tugas Akhir/cleaned data/annot/test/c10_s7.csv') #check imbalance data
    'Check if data is imbalance or not'
    count_a = np.count_nonzero(annot == 1)
    print('Number of As in annot: ', count_a)
    count_n = np.count_nonzero(annot == 0)
    print('Number of Ns in annot: ', count_n)
    

    
    'Read balanced labels dataset'
    annot_map = annot.to_numpy()
    # annot_map = np.where(annot_map == 'N', 0,annot_map)
    # annot_map = np.where(annot_map == 'A', 1,annot_map)
    
    annot_map = np.array(annot_map)
    
    train(df_final_scaled, annot_map)
    model = dl_algo(X_train, y_train, X_test, y_test, X_val, y_val)
    model.summary()

    
    'Plot'
    train_data(X_train, y_train, X_test, y_test, X_val, y_val, model)


def train(df_2_scaled, annot_map):
    print('Begin train funct')
    global X_train, y_train
    global X_test, y_test
    global X_val, y_val
    global train_x_scaled,test_x_scaled 
    
    print('data input shape before train-test-split: ',df_2_scaled.shape)
    print('annot input shape before train-test-split: ',annot_map.shape)
    X_train, X_test, y_train, y_test = train_test_split(df_2_scaled, annot_map, test_size=0.2,stratify=np.array(annot_map),random_state=10)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,stratify=np.array(y_test),random_state=10)
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
 
   
   


def dl_algo(X_train, y_train, X_test, y_test, X_val, y_val):
    
    
    input_ = Input(shape = X_train.shape[1:])
    filt_size = 60
    # x = Conv1D(filters=45, kernel_size=32, padding='same', kernel_initializer='he_normal',activation ='relu', name='block1_conv1') (input_)
    print('Begin computing block no: 1')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block1_conv') (input_)
    x = tf.keras.layers.BatchNormalization(name='block1_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block1_MaxPool') (x)
    x = Dropout(0.5,name = 'block1_DropOut') (x)  
    print('Finish computing block no: 1')
    
    print('Begin computing block no: 2')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block2_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block2_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block2_MaxPool') (x)
    x = Dropout(0.5,name = 'block2_DropOut') (x)  
    print('Finish computing block no: 2')
    
    
    print('Begin computing block no: 3')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block3_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block3_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block3_MaxPool') (x)
    x = Dropout(0.5,name = 'block3_DropOut') (x)  
    print('Finish computing block no: 3')
    
    
    print('Begin computing block no: 4')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block4_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block4_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block4_MaxPool') (x)
    x = Dropout(0.5,name = 'block4_DropOut') (x)  
    print('Finish computing block no: 4')
    
    
    
    print('Begin computing block no: 5')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block5_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block5_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block5_MaxPool') (x)
    x = Dropout(0.5,name = 'block5_DropOut') (x)  
    print('Finish computing block no: 5')
    
    
    print('Begin computing block no: 6')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block6_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block6_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block6_MaxPool') (x)
    x = Dropout(0.5,name = 'block6_DropOut') (x)  
    print('Finish computing block no: 6')
    
    
    print('Begin computing block no: 7')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block7_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block7_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block7_MaxPool') (x)
    x = Dropout(0.5,name = 'block7_DropOut') (x)  
    print('Finish computing block no: 7')
    
    
    
    print('Begin computing block no: 8')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block8_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block8_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block8_MaxPool') (x)
    x = Dropout(0.5,name = 'block8_DropOut') (x)  
    print('Finish computing block no: 8')
    
    
    
    print('Begin computing block no: 9')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block9_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block9_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block9_MaxPool') (x)
    x = Dropout(0.5,name = 'block9_DropOut') (x)  
    print('Finish computing block no: 9')
    
    
    print('Begin computing block no: 10')
    x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block10_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block10_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block10_MaxPool') (x)
    x = Dropout(0.5,name = 'block10_DropOut') (x)  
    print('Finish computing block no: 10')
    
    # print('Begin computing block no: 11')
    # x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block11_conv') (x)
    # x = tf.keras.layers.BatchNormalization(name='block11_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = MaxPool1D(pool_size=2, strides=2, name = 'block11_MaxPool') (x)
    # x = Dropout(0.5,name = 'block11_DropOut') (x)  
    # print('Finish computing block no: 11')
    
    # print('Begin computing block no: 12')
    # x = Conv1D(filters=filt_size, kernel_size=32, padding='same', kernel_initializer='he_normal', name='block12_conv') (x)
    # x = tf.keras.layers.BatchNormalization(name='block12_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = MaxPool1D(pool_size=2, strides=2, name = 'block12_MaxPool') (x)
    # x = Dropout(0.5,name = 'block12_DropOut') (x)  
    # print('Finish computing block no: 12')
    
    
    'Flatten before FC'
    x = Flatten(name='flatten') (x)
    print('Finish flattening')
    'Classification Layer'
    
    print('Begin computing Classification Layer-1')
    x = Dense(512, kernel_initializer='he_normal', name='fc1_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc1_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc1_DropOut') (x) 
    print('Finish computing Classification Layer-1')
    
    print('Begin computing Classification Layer-2')
    x = Dense(512, kernel_initializer='he_normal', name='fc2_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc2_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc2_DropOut') (x) 
    print('Finish computing Classification Layer-2')
    
    print('Begin computing Classification Layer-3')
    x = Dense(512, kernel_initializer='he_normal', name='fc3_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc3_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc3_DropOut') (x) 
    print('Finish computing Classification Layer-3')
    
    print('Begin computing Classification Layer-4')
    x = Dense(512, kernel_initializer='he_normal', name='fc4_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc4_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc4_DropOut') (x) 
    print('Finish computing Classification Layer-4')
    
    print('Begin computing Classification Layer-5')
    x = Dense(512, kernel_initializer='he_normal', name='fc5_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc5_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc5_DropOut') (x) 
    print('Finish computing Classification Layer-5')
    
    print('Begin computing Classification Layer-6')
    x = Dense(512, kernel_initializer='he_normal', name='fc6_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc6_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(0.5,name = 'fc6_DropOut') (x) 
    print('Finish computing Classification Layer-6')
    'Softmax Output'
    # x = Dense(2,activation='softmax', name='out')(x)
    x = Dense(1,activation='sigmoid', name='out')(x)
    model = Model(input_, x)
    
    return model

def train_data(X_train, y_train, X_test, y_test, X_val, y_val, model):
    global epoch_list,y_train_acc, my_predict, my_predict_2
    'convert to tensor'
  
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()

    
    cb_dir = 'DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.h5'
    cb_dir_csv = 'DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.csv'
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(cb_dir,save_best_only=True)
    checkpoint_cb_csv =  tf.keras.callbacks.CSVLogger(cb_dir_csv)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=30,restore_best_weights=True)
    
    
    model_dir ='DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_model_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.h5'
    model_arch_dir = 'DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_arch_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.png'
    
    metrics_dir = 'DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_metrics_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.csv'
    
    #fit the model - Train Data
    history = model.fit(x=X_train,y=y_train,epochs=200,validation_data=(X_val,y_val),callbacks=[checkpoint_cb,early_stopping_cb,checkpoint_cb_csv])
    model.save(model_dir)
    model.summary()
    plot_model(model, to_file=model_arch_dir, show_shapes=True)
    my_predict = model.predict(X_test) #use new train model
    
    'use this if u wanna load trained data...'
    # saved_model = tf.keras.models.load_model(model_dir)
    # my_predict = saved_model.predict(X_test) #use saved_model train file
    
    
    print('mypredict_ :  ' , my_predict)
    
    my_predict_2 = my_predict.copy()
    my_predict_2[my_predict_2>=0.5]=1
    my_predict_2[my_predict_2<0.5]=0
    
    # my_predict_2
    
    accuracy = accuracy_score(y_test, my_predict_2)
    precision = precision_score(y_test, my_predict_2)
    recall = recall_score(y_test, my_predict_2)
    f1 =  f1_score(y_test, my_predict_2)
    
    accuracy = (accuracy*100)
    precision= (precision*100)
    recall= (recall*100)
    f1 = (f1*100)
    print('Accuracy: %.3f' % accuracy + '%')
    print('Precision:  %.3f' % precision+ '%' )
    print('Recall:  %.3f' % recall+'%'  )
    print('F1 Score:  %.3f' %f1 +'%' )
    
    sensitivity = imblearn.metrics.sensitivity_score(y_test, my_predict_2)
    sensitivity = (sensitivity*100)
    specificity = imblearn.metrics.specificity_score(y_test, my_predict_2)
    specificity = (specificity*100)
    print('Sensitivity: %.3f ' % sensitivity + '%')
    print('Specificity: %.3f  ' % specificity + '%')
    
    metrics = []
    metrics = [accuracy,precision,recall,f1,sensitivity,specificity]
    metrics = pd.DataFrame([metrics])
    metrics.columns=['Accuracy','Precision','Recall','F1_Score','Sensitivity','Specificity']
    
   
    #save metrics report to csv
    print('Saving metrics to: '+metrics_dir)
    metrics.to_csv(metrics_dir,index=False)
    

    
    confusion_matrix_sk = confusion_matrix(y_test, my_predict_2)
    sns.heatmap(confusion_matrix_sk, annot=True, fmt="d");
    
    plt.xlabel("Predicted Value");
    plt.ylabel("True Value");
    plt.savefig('DL/balanced/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'confmatrix_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.png')
    
   
    return 

    
if __name__ == "__main__":
    start = timer()
    main() 
    end = timer()
    print('Time elapsed: ', timedelta(seconds=end-start))



