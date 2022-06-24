# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 08:44:48 2021

@author: alani
"""

import pandas as pd
import matplotlib.pyplot as plt
'99: kernel size 16'

'architecture parameters'
imf_no = '2'
test_no = '29'
conv_no = '10'
dense_no = '4'
filt_no ='45'

cb_dir = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.h5'
cb_dir_csv = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_cb_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.csv'
val_fig_dir = 'DL/filt'+filt_no+'/imf'+imf_no+'/'+test_no+'_valmat_imf'+imf_no+'_conv'+conv_no+'_d'+dense_no+'.png'


'Metrics of Training-Validation w/ ratio 80 10 10'
df_epoch = pd.read_csv(cb_dir_csv)
acc = df_epoch.loc[:,'accuracy'].tolist()
val_acc = df_epoch.loc[:,'val_accuracy'].tolist()
loss = df_epoch.loc[:,'loss'].tolist()
val_loss = df_epoch.loc[:,'val_loss'].tolist()



epoch_list = list(range(1,36)) #EPOCH=150
y_train_acc = acc
y_val_acc = val_acc
y_train_loss = loss
y_val_loss = val_loss

f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
t = f.suptitle('IMF'+imf_no+' 1DCNN Conv'+conv_no+' Dense'+dense_no+', filt'+filt_no+'',fontsize=9)

ax1.plot(epoch_list,y_train_acc,label='Train Accuracy')
ax1.plot(epoch_list,y_val_acc,label='Validation Accuracy')
#ax1.set_xticks(np.arange(0,12,1))
#ax1.set_ylim(0.75,0.85)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1=ax1.legend(loc="best")

ax2.plot(epoch_list,y_train_loss,label='Train Loss')
ax2.plot(epoch_list,y_val_loss,label='Validation Loss')
#ax2.set_xticks(np.arange(0,12,1))
#ax2.set_ylim(0,1)
ax2.set_ylabel('Cross Entropy')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2=ax2.legend(loc="best")
plt.savefig(val_fig_dir)
print('Validation saved to: ', val_fig_dir)