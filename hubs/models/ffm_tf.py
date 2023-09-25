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

##import the metrics libraries
from sklearn.metrics import accuracy_score as acs

##for denormalizing the data
from sklearn.preprocessing import StandardScaler


class ffm_tf:
    def __init__(self):
        pass


    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition, chk_name, train, original_features, original_labels, neurons):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alfa)

        ##lets make an stop function
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)

        if train:
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

            ##SKLEARN for accuracy metrics
            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Accuracy: {accuracy:.2f}%')

            ##lets ask if the user wants to store the model
            r = input('Savel model? (Y-N)')
            if r == 'Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
                print(f'Checkpoint path: {checkpoint_file}')
                model.save(checkpoint_file)
                print('Model Saved!')

            elif r == 'N':
                print('Model NOT saved')

            else:
                print('Command not recognized')

        else:
            ##
            ##we are not training a model here, just using an already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
            ##lets load the model
            keras.models.load_model(checkpoint_file)

            ##prediction output
            pred_out = model.predict(train_features)

            ##lets denormalize the data
            SC = StandardScaler()

            original_labels_norm = SC.fit_transform(original_labels)
            
            if neurons == 1:
                pred_out = pred_out.reshape(-1,1)
            
            pred_out_denorm = SC.inverse_transform(pred_out)

            pred_df = pd.DataFrame(pred_out_denorm)

            result_data = pd.concat([original_features, pred_df], axis=1)

            print(f'Dataframe: {result_data}')


            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'results'))

            #print(results_dir)

            results_file = os.path.join(results_dir, f'{chk_name}_RESULTS_ffm.xlsx')

            ##original_features.to_excel('output.xlsx', index=False, engine='openpyxl')

            ##lets store the dataframe as excel file
            result_data.to_excel(results_file, index=False, engine='openpyxl')


            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(train_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title(f'Prediction Output of model {chk_name}')
            plt.legend()
            plt.show()

            ##lets show the accuracy value for this training batch
            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction accuracy: {accuracy:.2f}%')


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








