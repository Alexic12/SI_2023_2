import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

##commom modules
import numpy as np
import pandas as pd
import os
import sys

## import tools for visualization
import matplotlib.pyplot as plt

class ffm_tf:
    def __init__(self) -> None:
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        model = self.build_model(train_features.shape[1]+1, train_labels.shape[1], alfa)

        ##lets make an stop function
        early_stop = keras.callbacks.EarlyStopping(monitor = 'mse', patience=stop_condition)
        
        ##lets train the model
        history = model.fit(train_features, train_labels, epochs = iter, verbose = 1, callbacks=[early_stop], validation_split=0)

        print(history)

        training_data = pd.DataFrame(history.history)

        print(training_data)

        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()

        ##lets validate the trained model

        pred_out = model.predict(test_features)

        plt.figure()
        plt.plot(pred_out, 'r', label="Prediction_Output")
        plt.plot(test_labels, 'b', label="Real Output")
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
            Dense(output, activation=tf.nn.sigmoid)
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss='mse', metrics=['mse'])

        return model