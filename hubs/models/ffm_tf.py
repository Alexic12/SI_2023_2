#lets import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt

class ffm_tf:
        def __init__(self):
            pass
        
        def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
              model = self.build_model(train_features.shape[1]+1, train_labels.shape[1], alfa)

              #lets make an stop function
              early_stop= keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

              history = model.fit(train_features, train_labels, epochs=500, verbose = 1, callbacks = [early_stop], validation_split=0)
              print(history)



        def build_model(self, layers, hidden_neurons, output, alfa):
              model = keras.Sequential([
                    Dense(hidden_neurons, activation = tf.nn.sigmoid, input_shape=[hidden_neurons-1]),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(hidden_neurons, activation = tf.nn.sigmoid),
                    Dense(output)
              ])

              model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss= 'mse', metrics=['mse'])

              return model