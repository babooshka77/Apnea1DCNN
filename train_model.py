import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import imblearn
import seaborn as sns

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
'Checkpoint Callback'
'Jika lebih dari patience dan tidak ada perubahan signifikan, data train stop'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("DL/tes_model.h5",save_best_only=True)
checkpoint_cb_csv =  tf.keras.callbacks.CSVLogger("DL/tes_model.csv")
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

#fit the model - Train Data
history = model.fit(x=X_train,y=y_train,epochs=100,validation_data=(X_val,y_val),callbacks=[checkpoint_cb,early_stopping_cb,checkpoint_cb_csv])
model.save("DL/tes_model.h5")

'use this if u wanna load trained data...'
# saved_model = tf.keras.models.load_model('DL/tes_model.h5')

'Predict your data using test data'
my_predict = model.predict(X_test)
print('mypredict_ :  ' , my_predict)

my_predict_2 = my_predict.copy()
my_predict_2[my_predict_2>=0.5]=1
my_predict_2[my_predict_2<0.5]=0
np.savetxt('my_predict_2.csv', my_predict_2, delimiter=',')

'Plot metrics'
accuracy = accuracy_score(y_test, my_predict_2)
precision = precision_score(y_test, my_predict_2)
recall = recall_score(y_test, my_predict_2)
f1 =  f1_score(y_test, my_predict_2)
sensitivity = imblearn.metrics.sensitivity_score(y_test, my_predict_2)
specificity = imblearn.metrics.specificity_score(y_test, my_predict_2)
print('Accuracy: %.3f ' % (accuracy*100))
print('Precision: %.3f  ' % (precision*100))
print('Recall: %.3f  ' % (recall*100))
print('F1 Score: %.3f  ' % (f1*100))
print('Sensitivity: %.3f ' % (sensitivity*100))
print('Specificity: %.3f  ' % (specificity*100))