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
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition,chk_name,train):
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alfa)

        ##lets make an stop function
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)

        if train:

            ##lets train the model
            history = model.fit(train_features, train_labels, epochs=iter, verbose = 1, callbacks = [early_stop], validation_split=0)

    

            training_data = pd.DataFrame(history.history)
            #print(training_data)
            #historial de prediccion
            plt.figure()
            plt.plot(training_data['mse'], 'r', label='mse')
            plt.show()

            ##lets validate the trained model
            #validacion de prediccion y comparacion con el modelo real

            pred_out = model.predict(test_features)

            plt.figure()
            plt.plot(pred_out, 'r', label='Prediction Output') #muestra lo que predice 
            plt.plot(test_labels, 'b', label='Real Output') #muestra lo que se supone que deberia ser el resultado
            plt.show()


            ##SKLEARN for accuracy metrics
            #mostrar la precision del entrenamiento
            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100 #convierte a entero, se multiplica por 100 porque el sistema de el resultado en porcentaje
            print(f'Accuracy: {accuracy:.2f}%') #se esta diciendo que es un float con dos decimales

        
            ## GUARDAR EL MODELO - TAREA 1

            r = input('Save model? (Y-N)')
                #Cuando se ha entrenado el modelo, brinda la opcion de guardarlo o no
            if r == 'Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm'))
        
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
                print(f'Checkpoint path: {checkpoint_file}')
                model.save(checkpoint_file) #En tensorflow es model.save .. En xgboost es model.save_model
                print('Model Saved!')

            elif r == 'N':
                print('Model NOT saved')

            else:
                print('Command not recognized')
        else:
            
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm')) 
            #los checkpoints se crean cada vez que se hace un nuevo entrenamiento
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
            ##lets load the model

            ## PREGUNTAR SI ASI ESTA BIEN CARGADO EL MODELO ?????
            tf.keras.models.load_model(checkpoint_file) #carga el archivo .h5 con el valor numeroco de los pesos y lo ubica en el modelo ya existente 

            ##prediction output
            pred_out = model.predict(train_features) 

            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(test_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title('Validation')
            plt.legend()
            plt.show()

            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100 #se pone train features para entrenar con la mayor parte de la data
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








