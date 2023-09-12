##lets import libraries 
import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense
##common libraries 
import numpy as np 
import pandas as pd 
import os 
import sys 
##import tools for data visualization 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class ffm_tf:
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels, test_labels, iter, alpha, stop_condition):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alpha)
        
        ##lets maje an stop function 
        
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)
        
        ##lets train the model 
        
        history = model.fit(train_features, train_labels, epochs=iter, verbose=1, callbacks=[early_stop], validation_split=0)
        
        training_data = pd.DataFrame(history.history)
        
        print(training_data)
        
        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()
        
        ##lets validate the trained model
        
        pred_out = model.predict(test_features)
        scaler = StandardScaler()
        data_labels_norm = scaler.fit_transform(pred_out)
        
        plt.figure()
        plt.plot(data_labels_norm, 'r', label='Prediction output')
        plt.plot(test_labels, 'b', label='Real output')
        plt.show()
        
        
        
    
    def build_model(self,hidden_neurons,output,alpha):
        model = keras.Sequential([
            Dense(hidden_neurons, activation=tf.nn.sigmoid, input_shape=[hidden_neurons-1]),   
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output,activation=tf.nn.sigmoid),
            Dense(output)
            ])
        
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha),loss='mse',metrics=['mse'])
        
        return model