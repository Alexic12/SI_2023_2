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


class ffm_tf:
    def _init_(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition,chk_name,train,outputs):
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

            ##Asking if the user wants to store the model
            r = input("Save Model? : ")
            if r == "Y":
                    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
                    checkpoint_file = os.path.join(model_dir, f"{chk_name}.h5")
                    model.save_model(checkpoint_file)
            elif r == "N":
                    print("Model NOT Saved")

            else:
                    print("Command not recognized")
        else: 
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
            checkpoint_file = os.path.join(model_dir, f"{chk_name}.json")
            model.load_model(checkpoint_file)

            pred_out = model.predict(train_features)

            sc = StandardScaler()
            ##Denormalizing data
            original_labels_norm= sc.fit_transform(original_labels)

            if outputs == 1:
                pred_out = pred_out.reshape(-1,1)

            pred_out_denorm = sc.inverse_transform(pred_out)
            
            pred_df = pd.DataFrame(pred_out_denorm)
            result_data=pd.concat([original_features, pred_df], axis = 1)

            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'results'))
            
            plt.figure()
            plt.plot(pred_out, "r", label="Model Output")
            plt.plot(train_labels, "b", label="Real Output")
            plt.xlabel("data points")
            plt.ylabel("Validation")
            plt.title("Prediction Output of model {chk_name}")
            plt.legend()
            plt.show()
            
            ##Accuracy

            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction Accuracy: {accuracy:.2f}%')
            

            
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