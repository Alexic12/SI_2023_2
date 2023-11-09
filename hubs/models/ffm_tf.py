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

## import scikit-learn to the metricks
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
                                                                    # si ponemos el verbose 1 el muestra en terminal esta entrenando donde esta entrenando y que iteracion va 
                                                                    #si ponemos un 0 no dice nada solo esta ejecutando.
        

            training_data = pd.DataFrame(history.history)
            #print(training_data) show the proces 

            plt.figure()
            plt.plot(training_data['mse'], 'r', label='mse')
            plt.show()


            ##lets validate the trained model este es la validacion  y ploteamos con la verdadera

            pred_out = model.predict(test_features)

            plt.figure()
            plt.plot(pred_out, 'r', label='Prediction Output')
            plt.plot(test_labels, 'b', label='Real Output')
            plt.show()

            ##Skelearn  metric libraris , plot library , help neural models 

            accuracy = acs(test_labels.astype(int),pred_out.astype(int)) *100 # predicion y el test comparelo que salio y lo que deberia toca convetir en int

            print(f'Accuracy:{accuracy:2f}%') # vamos a ver el entrenamiento de hisotory y los datos de validacion podemos ver la exactitu del modelo
            ## lets ask if the user wants yo store the model
            r=input('savel model?(Y-N)')
            if r=='Y':
                    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf')) # esto es para definir donde va la informacion para todos loslugares  toma la carpta donde esta
                    checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5') # es el nombre del modelo se pondra en otro parametro 
                    #checkpoint_file = os.path.join(model_dir, f'{chk_name}.json')
                    print(f'Checkpoint path: {checkpoint_file}')
                    model.save_weights(checkpoint_file)
                    #model.save(checkpoint_file)
                    print('Model Saved!')

            elif r == 'N':
                print('Model NOT Saved!')

            else:
                print('Command not recognized')
                    # solo los el json, tiene la fonciguraion de entradas de los modelos que usamos o info


        else:
            ##we are not training a model here, just using an already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf')) # para saber donde esta el modelo 
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5') # con esto vamos a cargar el modelo o cual vamos a cargar.
            ##lets load the model
            model.load_weights(checkpoint_file) # los valores numericos. 
            #model.load(checkpoint_file) 
            # ahora vamos a predecir los valores de  neuvo 
            ##prediction output
            pred_out = model.predict(train_features)# con todos los valores 
            
            #data = pd.DataFrame(train_features)

            
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
        # lost es error cuadratico medio
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=alfa), loss='mse', metrics=['mse'])

        return model