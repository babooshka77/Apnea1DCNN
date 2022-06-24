# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:43:37 2021

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
    global test_no, conv_no, dense_no, filt_no,k_size,dropout_rate,k_init 
    global filt_size

    imf_no = '2'
    test_no = '22'
    conv_no = '10'
    dense_no = '4'
    filt_no ='45'
    
    filt_size = 45
    k_size =32
    dropout_rate = 0.5 
    global dense_neuron 
    dense_neuron = 512
    k_init = 'he_normal'
    print('Begin main funct')
    # recordAnnot_imf1_final
    # df_final_dir = 'E:/Felix/1822041/Tugas Akhir/new/imf_'+imf_no+'/recordAnnot_imf'+imf_no+'_final.csv'
    df_final_dir = 'E:/Felix/1822041/Tugas Akhir/data_imfs_rev/imf_'+imf_no+'/record_imf'+imf_no+'_final.csv'
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
    annot_dir = 'E:/Felix/1822041/Tugas Akhir/annot/annot.csv'
    annot = pd.read_csv(annot_dir)
    print('Annotation shape: ', annot.shape)
   
    
    count_a = np.count_nonzero(annot == 'A')
    print('Number of As in annot: ', count_a)
    count_n = np.count_nonzero(annot == 'N')
    print('Number of Ns in annot: ', count_n)
    
 
    'Change Labels from A/N to 1/0, where A = 1, N = 0'
    annot_map = annot.to_numpy()
    annot_map = np.where(annot_map == 'N', 0,annot_map)
    annot_map = np.where(annot_map == 'A', 1,annot_map)
    
    annot_map = np.array(annot_map)
    
    train(df_final_scaled, annot_map)
    model = dl_algo(X_train, y_train, X_test, y_test, X_val, y_val)
    model.summary()

    'Plot Model'

    
    'Plot'
    train_data(X_train, y_train, X_test, y_test, X_val, y_val, model)


def train(df_2_scaled, annot_map):
    print('Begin train funct')
    global X_train, y_train
    global X_test, y_test
    global X_val, y_val
    global train_x_scaled,test_x_scaled 
    
    'Train-Test-Split data into Train, Test, Valid datasets with random state 10'
    print('data input shape before train-test-split: ',df_2_scaled.shape)
    print('annot input shape before train-test-split: ',annot_map.shape)
    X_train, X_test, y_train, y_test = train_test_split(df_2_scaled, annot_map, test_size=0.4,stratify=np.array(annot_map),random_state=10)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,stratify=np.array(y_test),random_state=10)
   
    'array values in each portion of data is set to type float32'
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
    
    'Training Block'
    'Feature Extraction Layer starts here'
    # x = Conv1D(filters=45, kernel_size=32, padding='same', kernel_initializer='he_normal',activation ='relu', name='block1_conv1') (input_)
    print('Begin computing block no: 1')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block1_conv') (input_)
    x = tf.keras.layers.BatchNormalization(name='block1_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block1_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block1_DropOut') (x)  
    print('Finish computing block no: 1')
    
    print('Begin computing block no: 2')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block2_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block2_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block2_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block2_DropOut') (x)  
    print('Finish computing block no: 2')
    
    
    print('Begin computing block no: 3')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block3_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block3_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block3_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block3_DropOut') (x)  
    print('Finish computing block no: 3')
    
    
    print('Begin computing block no: 4')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block4_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block4_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block4_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block4_DropOut') (x)  
    print('Finish computing block no: 4')
    
    
    
    print('Begin computing block no: 5')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block5_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block5_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block5_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block5_DropOut') (x)  
    print('Finish computing block no: 5')
    
    
    print('Begin computing block no: 6')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block6_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block6_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block6_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block6_DropOut') (x)  
    print('Finish computing block no: 6')
    
    
    print('Begin computing block no: 7')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block7_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block7_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block7_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block7_DropOut') (x)  
    print('Finish computing block no: 7')
    
    
    
    print('Begin computing block no: 8')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block8_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block8_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block8_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block8_DropOut') (x)  
    print('Finish computing block no: 8')
    
    
    
    print('Begin computing block no: 9')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block9_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block9_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block9_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block9_DropOut') (x)  
    print('Finish computing block no: 9')
    
    
    print('Begin computing block no: 10')
    x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block10_conv') (x)
    x = tf.keras.layers.BatchNormalization(name='block10_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = MaxPool1D(pool_size=2, strides=2, name = 'block10_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block10_DropOut') (x)  
    print('Finish computing block no: 10')
    
    # print('Begin computing block no: 11')
    # x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block11_conv') (x)
    # x = tf.keras.layers.BatchNormalization(name='block11_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = MaxPool1D(pool_size=2, strides=2, name = 'block11_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block11_DropOut') (x)  
    # print('Finish computing block no: 11')
    
    # print('Begin computing block no: 12')
    # x = Conv1D(filters=filt_size, kernel_size=k_size, padding='same', kernel_initializer=k_init, name='block12_conv') (x)
    # x = tf.keras.layers.BatchNormalization(name='block12_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = MaxPool1D(pool_size=2, strides=2, name = 'block12_MaxPool') (x)
    # x = Dropout(dropout_rate,name = 'block12_DropOut') (x)  
    # print('Finish computing block no: 12')
    
    'Feature Extraction Layer ends here'

    'Flatten before FC'
    x = Flatten(name='flatten') (x)
    print('Finish flattening')

    'Classification Layer starts here'
    
    print('Begin computing Classification Layer-1')
    x = Dense(dense_neuron, kernel_initializer=k_init, name='fc1_dense' )(x)
    x = tf.keras.layers.BatchNormalization(name='fc1_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(dropout_rate,name = 'fc1_DropOut') (x) 
    print('Finish computing Classification Layer-1')
    
    print('Begin computing Classification Layer-2')
    x = Dense(dense_neuron, kernel_initializer=k_init, name='fc2_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc2_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(dropout_rate,name = 'fc2_DropOut') (x) 
    print('Finish computing Classification Layer-2')
    
    print('Begin computing Classification Layer-3')
    x = Dense(dense_neuron, kernel_initializer=k_init, name='fc3_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc3_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(dropout_rate,name = 'fc3_DropOut') (x) 
    print('Finish computing Classification Layer-3')
    
    print('Begin computing Classification Layer-4')
    x = Dense(dense_neuron, kernel_initializer=k_init, name='fc4_dense')(x)
    x = tf.keras.layers.BatchNormalization(name='fc4_BatchNorm') (x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    x = Dropout(dropout_rate,name = 'fc4_DropOut') (x) 
    print('Finish computing Classification Layer-4')
    
    # print('Begin computing Classification Layer-5')
    # x = Dense(512, kernel_initializer='he_normal', name='fc5_dense')(x)
    # x = tf.keras.layers.BatchNormalization(name='fc5_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = Dropout(0.5,name = 'fc5_DropOut') (x) 
    # print('Finish computing Classification Layer-5')
    
    # print('Begin computing Classification Layer-6')
    # x = Dense(512, kernel_initializer='he_normal', name='fc6_dense')(x)
    # x = tf.keras.layers.BatchNormalization(name='fc6_BatchNorm') (x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu) (x)
    # x = Dropout(0.5,name = 'fc6_DropOut') (x) 
    # print('Finish computing Classification Layer-6')
    'Classification Layer ends here'


    'Output Layer'
    'softmax'
    # x = Dense(2,activation='softmax', name='out')(x)
    'sigmoid'
    x = Dense(1,activation='sigmoid', name='out')(x)
    model = Model(input_, x)
    
    return model

def train_data(X_train, y_train, X_test, y_test, X_val, y_val, model):
    global epoch_list,y_train_acc, my_predict, my_predict_2

    'compile model with crossentropy loss function and adam optimizer'
    model.compile(loss='crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()
    
    'callback directory'
    cb_dir = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.h5'
    cb_dir_csv = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.csv'
    
    'callbakcs'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(cb_dir,save_best_only=True)
    checkpoint_cb_csv =  tf.keras.callbacks.CSVLogger(cb_dir_csv)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)
    
    
    model_dir ='DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_model_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.h5'
    model_arch_dir = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_arch_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.png'
    
    metrics_dir = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_metrics_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.csv'
    
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
    # ' use if SOFTMAX'
    # global output_my_predict_2 
    # my_predict_2 = pd.DataFrame(my_predict, columns=['Normal','Apnea'])
    # output_my_predict_2 = pd.DataFrame([])
    # output_my_predict_2 = np.where(my_predict_2['Normal']>=my_predict_2['Apnea'],0,1)


    'Data test scores with SIGMOID function'
    my_predict_2 = my_predict.copy()
    my_predict_2[my_predict_2>=0.5]=1
    my_predict_2[my_predict_2<0.5]=0

    
    accuracy = accuracy_score(y_test, my_predict_2)
    precision = precision_score(y_test, my_predict_2)
    recall = recall_score(y_test, my_predict_2)
    f1 =  f1_score(y_test, my_predict_2)
    
    accuracy = (accuracy*100)
    precision= (precision*100)
    recall= (recall*100)
    f1 = (f1*100)
    sensitivity = imblearn.metrics.sensitivity_score(y_test, my_predict_2)
    sensitivity = (sensitivity*100)
    specificity = imblearn.metrics.specificity_score(y_test, my_predict_2)
    specificity = (specificity*100)
    print('Accuracy: %.3f' % accuracy + '%')
    print('Precision:  %.3f' % precision+ '%' )
    print('Recall:  %.3f' % recall+'%'  )
    print('F1 Score:  %.3f' %f1 +'%' )  
    print('Sensitivity: %.3f ' % sensitivity + '%')
    print('Specificity: %.3f  ' % specificity + '%')
    
    metrics = []
    metrics = [accuracy,precision,recall,f1,sensitivity,specificity]
    metrics = pd.DataFrame([metrics])
    metrics.columns=['Accuracy','Precision','Recall','F1_Score','Sensitivity','Specificity']
    
   
    #save metrics report to csv
    print('Saving metrics to: '+metrics_dir)
    metrics.to_csv(metrics_dir,index=False)
    
    # report = metrics.classification_report(my_predict_2, output_dict=True)
    # df_report = pd.DataFrame(report).transpose()
    # save_df_report = df_report.to_csv("DL/filt60/imf"+imf_no+"/metrics_train_result_imf"+imf_no+"_e100_test_"+test_no+'cov'+conv_no+'_de_'+ dense_no+".csv")
    
    'plot confusion matrix'
    confusion_matrix_sk = confusion_matrix(y_test, my_predict_2)
    sns.heatmap(confusion_matrix_sk, annot=True, fmt="d");
    
    plt.xlabel("Predicted Value");
    plt.ylabel("True Value");
    plt.savefig('DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'confmatrix_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.png')
 
    return 


    
if __name__ == "__main__":
    start = timer()
    main() 
    end = timer()
    print('Time elapsed: ', timedelta(seconds=end-start))


