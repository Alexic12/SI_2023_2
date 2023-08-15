import os
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection #dividir la tabla en entrenamiento y test
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

#300 for test
#2200 for train
#for data base with 2500 dates

class Data:
    """
    Attributes:

    methods:

    """
    def __init__(self):
        pass

    def data_process(self, file, test_split, norm, neurons):
        #file = database rute file
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        ##Lets confert the raw data to an array
        data_arr = np.array(data_raw)

        ##Lets label enconde any text in the data

        ##Lets create a boolean array with the size of the columns
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        ##lets read columns data type
        for i in range(0, data_arr.shape[1]):
        #Vamos detectar si una columna tiene un string o un numero por medio de un booleano

            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
              #Mira si la columna booleana que creamos en int o String
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:,i]) + 1

        ##Lets split the data into features and labels
        ##features are inputs, labels are output
          #Diferenciar los features de los labels
        data_features = data_arr[:,0:-neurons]
        data_labels = data_arr[:, -neurons:]

        print(f'DATA-FEATURES: {data_features}')
        print(f'DATA-LABELS: {data_labels}')

        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)

        ##lets check the dimensions of the arrays
        print(f'Dimensions: {data_labels.shape}')

        #Los labels son la salida, son el calculo que nos debe dar con las features
        if norm == True:

            ##Lets normalize the data
            scaler = StandardScaler()##Create an object of this library in 
            #Crear un objeto de esta libreria en particular

            data_features_norm = scaler.fit_transform(data_features) #Normalizando las featurs (input)
            data_labels_norm = scaler.fit_transform(data_labels)  #Nomalizar los labels (output)

        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        print(data_labels_norm)
        
        #lets split the data into training and testing
        ##NO CAMBIAR EL ORDEN DEL TTS, SINO NO FUNCIONA
        ##input (train, test) output(train, test)
        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size= test_split)

        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'labels: {train_labels}')

        #destinamos el tamaño de la data para entrenamiento como el 10 por ciento


        return train_features, test_features, train_labels, test_labels