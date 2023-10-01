import os
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

import tensorflow
from tensorflow import keras


class Data:
    """
    Attributes:


    Methods:
    
    """
    def __init__(self):
        self.scaler = StandardScaler()##Create an object of this library in particular
        #necesitamos iniciarlo como un objeto global para ingresar las variables necesarias para desnormalizar los datos

    def data_process(self, file, test_split, norm, neurons, avoid_col):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        #print(f'Data_Raw {data_raw}')

        ##lets store the original features
        columns = data_raw.shape[1] #con esto tomamos la cantidad de columnas del dataframe
        original_features = data_raw[data_raw.columns[:columns-neurons]] #toma las columnas menos las salidas
        original_labels = data_raw[data_raw.columns[columns-neurons:columns]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##original_features.to_excel('output.xlsx', index=False, engine='openpyxl')

        ##Lets convert the raw data to an array
        data_arr = np.array(data_raw)

        ##Lets label encode any text in the data

        ##Lets create a boolean array with the size of the columns of the original data array
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        ##lets read columns data type
        for i in range(0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:,i]) + 1


        ##Lets split the data into features and labels
        data_features = data_arr[:,avoid_col:-neurons]
        data_labels = data_arr[:, -neurons:]

        print(f'DATA_FEATURES: {data_features}')
        print(f'DATA_LABELS: {data_labels}')
        
        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)

        ##lets check the dimensions of the arrays
        #print(f'Dimensions: {data_labels.shape}')

        if norm == True:
            ##Lets normalize the data
            #fit transform utiliza valores de los extremos para normalizar, el mayor y el menor
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)    
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'labels: {train_labels}')


        return train_features, test_features, train_labels, test_labels, original_features, original_labels
        #Entonces entrega el dataframe original y los datos de entrenamiento

    #Aqui se desnormaliza la data para que el usuario pueda utilizarla
    # y se ponen los atributos o variables necesarias para esta operacion
    def denormalize(self, data):

        data_denorm = self.scaler.inverse_transform(data)

        return data_denorm
    
    #Funcion para las bases de datos tomadas de repositorio de keras para imagenes 

    
    def download_database(self, database):
        #agregar repositorio de bases de datos (MNISt, CIFAR10,CIFAR100)
        if database == 'MNIST':
            #keras cargar la data
            #train images: imagenes de entrenamiento
            #train labels: Salidas
            #60000 imagenes de entrenamiento y 10000 de prueba
            (train_images, train_labels),(test_images, test_labels) = keras.datasets.mnist.load_data()

        elif database == 'CIFAR10':
            pass
        elif database == 'CIFAR100':
            pass

        return train_images, test_images, train_labels, test_labels 
    
    def timeseries_process(self, window_size, horizon_size, file, test_split, norm):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        ##lets turn it into an array
        array_raw = np.array(data_raw)

        ## Lets get the number of sample times for the whole data (# of sample times)
        data_length = array_raw.shape[1]

        print(f'Sample Times for time series: {data_length}')

        ##Lets create the data base array for storing the data the proper way
        ##Rows = (#Sample Times - Window - horizon)
        ##Columns = (window + horizon)
        time_series_arr = np.zeros((data_length - window_size - horizon_size + 1, window_size + horizon_size))

        print(f'Time_Series_Arr rows: {time_series_arr.shape[0]}')
        print(f'Data Array Raw Shape: {array_raw.shape}')
        print(f'Time_Series_Arr Shape: {time_series_arr.shape}')

        ##we have to look trough all the raw data, and take the correct data points and store them as window and horizon
        for i in range(data_length - window_size - horizon_size):
            time_series_arr[i] = array_raw[1, i:i+window_size+horizon_size]

        ##lets print this time_series_arr
        print('Time Series')
        print(time_series_arr)

        
        ##lets store the original features
        columns = time_series_arr.shape[1]
        original_features = data_raw[data_raw.columns[0:-horizon_size]]
        original_labels = data_raw[data_raw.columns[-horizon_size:]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##now that we have the time series as a normal database for neural networks we have to split it on features and labels
        data_features = time_series_arr[:, 0:-horizon_size]
        data_labels = time_series_arr[:, -horizon_size:]

        ##lets normalize the data
        
        if norm == True:
            ##Lets normalize the data
            if horizon_size == 1:
                data_labels = data_labels.reshape(-1,1)
            sc = StandardScaler()
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)    
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            #print(f'Features: {train_features}')
            #print(f'labels: {train_labels}')

        return train_features, test_features, train_labels, test_labels, original_features, original_labels


##Lets run this method of time_series just for testing
T = Data()
T.timeseries_process(3,1,'DATA_SENO_DIRECTO.xlsx')



        

