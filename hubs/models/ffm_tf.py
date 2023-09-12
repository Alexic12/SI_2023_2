#Lets import libriries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense #create layers with any neurons
##common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools
import matplotlib.pyplot as plt

class ffm_tf:
    def __init__(self):
        pass

    def run (self, train_features,test_features,train_labels,test_labels,iter,alfa, stop_condition):
        model= self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alfa)

        ##lets make an stop function
        early_stop =keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)

        ##lets train the model
        history = model.fit(train_features, train_labels, epochs=iter, verbose=1, callbacks= [early_stop], validation_split=0)#epochs=iteraciones

        print(history)

        training_data=pd.DataFrame(history.history)
        print(training_data)

        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()

        #lets validated the train model

        pred_out= model.predict(test_features)

        plt.figure()
        plt.plot(pred_out, 'r', label='prediction output')
        plt.plot(test_labels, 'b', label='Real Output')
        plt.show()

    def build_model(self,hidden_neurons, output,alfa):
        model=keras.Sequential([
            Dense(hidden_neurons, activation=tf.nn.sigmoid, input_shape=[hidden_neurons-1]),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(output)#A la salida no se pone activacion pa evitar ruid
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss='mse', metrics=['mse'])#learning_rate=taza de aprendizaje=alfa

        return model