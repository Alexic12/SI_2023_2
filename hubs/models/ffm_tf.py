##Lets import lobraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

import numpy as np
import pandas as pd
import os
import sys

##Import tools for data visualization
import matplotlib.pyplot as plt

class ffm_tf:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alfa)

        ##Lets make an stop function
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition) #Si el ecm no cambia en 20 iteraciones para el proceso

        ##Lets train the model
        history = model.fit(train_features, train_labels, epochs=iter, verbose = 1, callbacks=[early_stop], validation_split=0) #Epochs son las iteraciones, verbose=0 (no escribe nada en la consola)
        print(history)

        training_data = pd.DataFrame(history.history)
        print(training_data)

        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()

        #Lets validate the trained model
        pred_out = model.predict(test_features)
        plt.figure()
        plt.plot(pred_out, 'r', label='Prediction Outout')
        plt.plot(test_labels, 'b', label='Real output')
        plt.show()


    def build_model(self, hidden_neurons, output, alfa):
        model = keras.Sequential([
            Dense(hidden_neurons, activation=tf.nn.sigmoid, input_shape=[hidden_neurons-1]), #Capa de entrada
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(output) #Capa de salida

        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss='mse', metrics=['mse'])

        return model