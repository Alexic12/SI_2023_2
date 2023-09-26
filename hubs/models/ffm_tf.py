"""
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

import pandas as pd
import numpy as dp
import os
import sys

import matplotlib.pyplot as plt


class ffm_tf:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1],  alfa)

        ##Stop funtion
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=stop_condition)

        ##Traing
        history = model.fit(train_features, train_labels, epochs=iter, verbose= 1, callbacks = [early_stop], validation_split=0)
        print(history)


    def build_model(self, hidden_neurons, alfa):
        model = keras.Sequential([
            Dense(hidden_neurons, activation= tf.nn.sigmoid, input_shape= [hidden_neurons-1]),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(hidden_neurons, activation= tf.nn.sigmoid),
            Dense(output)


        ])

        model.compile(optimazer=keras.optimizers.Adam(learning_rate=alfa), loss="mse", metrics = ["mse"])

        return model
"""
##lets import libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

#from keras.layers.core import Dense

##common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for data visualization
import matplotlib.pyplot as plt


class ffm_tf:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alfa)

        ##lets make an stop function
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)

        ##lets train the model
        history = model.fit(train_features, train_labels, epochs=iter, verbose = 1, callbacks = [early_stop], validation_split=0)

    

        training_data = pd.DataFrame(history.history)
        #print(training_data)

        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()


        ##lets validate the trained model

        pred_out = model.predict(test_features)

        plt.figure()
        plt.plot(pred_out, 'r', label='Prediction Output')
        plt.plot(test_labels, 'b', label='Real Output')
        plt.show()









    def build_model(self, hidden_neurons, output, alfa):
        model = keras.Sequential([
            Dense(hidden_neurons, activation=tf.nn.sigmoid, input_shape=[hidden_neurons-1]),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(hidden_neurons, activation=tf.nn.sigmoid),
            Dense(output)

        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss='mse', metrics=['mse'])

        return model