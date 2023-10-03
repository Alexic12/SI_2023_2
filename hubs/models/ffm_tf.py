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
##import the metric libraries 
from sklearn.metrics import accuracy_score as acs
from hubs.data_hub import data

class ffm_tf:
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels, test_labels, original_feature, original_labels, iter, alpha, stop_condition, chk_name, train, neurons):
        
        model = self.build_model(train_features.shape[1] + 1, train_labels.shape[1], alpha)
        
        ##lets maje an stop function 
        
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=stop_condition)
        
        ##lets train the model 
        if train:
            
            history = model.fit(train_features, train_labels, epochs=iter, verbose=1, callbacks=[early_stop], validation_split=0)
            
            training_data = pd.DataFrame(history.history)
            
            print(training_data)
            
            plt.figure()
            plt.plot(training_data['mse'], 'r', label='mse')
            plt.show()
            
            ##lets validate the trained model
            
            pred_out = model.predict(test_features)
            #scaler = StandardScaler()
            #data_labels_norm = scaler.fit_transform(pred_out)
            
            plt.figure()
            plt.plot(pred_out, 'r', label='Prediction output')
            plt.plot(test_labels, 'b', label='Real output')
            plt.xlabel('Data points')
            plt.ylabel('normalize value')
            plt.title('Validation')
            plt.show()
            
            #accuracy metric 
            accurancy = acs(test_labels.astype(int),pred_out.astype(int))*100
            
            print(f'Accurancy: {accurancy:.2f}%')
            
            ##lets ask if the user wants to store the model 
            
            r = input('Save model? : (Y-N)')
            if r == 'Y':
                
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
                print(f'checkpoint path: {checkpoint_file}')
                model.save_weights(checkpoint_file)
                print('Model saved!')
                
            elif r == 'N':
                
                print('Model not saved!')
                
            else:
                
                print('Command not recognized')
            
        else:
            
            #we are not training a model her, just using a already existing model 
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'ffm_tf'))
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.h5')
            model.load_weights(checkpoint_file)
            
            ##Prediction output 
            pred_out = model.predict(train_features)
            
            ##lets denormalize the data 
            
            sc = StandardScaler()
            
            if neurons == 1:
                
                pred_out = pred_out.reshape(-1,1)
                
            pred_out_denorm = sc.inverse_transform(pred_out)
            
            pred_df = pd.DataFrame(pred_out_denorm)
            
            results_data = pd.concat([original_feature,pred_df], axis=1)
            
            print(f'DataFrame: {results_data}')
            
            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'results'))
            
            results_file = os.path.abspath(results_dir, f'{chk_name}_results_ffm_tf.xlsx')
            
            ##lets store dataframe as excel file 
            
            results_data.to_excel(results_file, index = False, engine = 'openpyxl')
            
            plt.figure()
            plt.plot(pred_out, 'r', label = 'Prediction output')
            plt.plot(train_labels, 'b', label = 'Real output')
            plt.xlabel('Data points')
            plt.ylabel('Normalize value')
            plt.title(f'Predict output of model {chk_name}')
            plt.show()
            
            #Accurancy metric 
            accurancy = acs(train_labels.astype(int),pred_out.astype(int))*100
            
            print(f'Prediction accurancy: {accurancy:.2f}%')
            
        
        
    
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