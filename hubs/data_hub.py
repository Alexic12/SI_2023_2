import os 
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler #Para normalizar los datos
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts #Parte los datos, con unos se entrena y con otros se testea


class Data:
    """
    Attribites: 

    Methods:
    """
    def __init__(self):
        pass
    
    def data_process(self, file, test_split, norm, neurons):
        #Lets define the absolute path for this folder
        dara_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","data"))
        
        #Find the complete excel file route
        excel_path = os.path.join(dara_dir, file)

        #Lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        #Lest convert the raw data to an array
        data_arr = np.array(data_raw)

        #Lets label encode any text in the data

        #Lets crate a boolean array with de size of the colms
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        #Lets read columns data type
        for i in range(0, data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:,i]) + 1
                 

        #Lets split the data into features=inputs and labels=outputs
        data_features = data_arr[:,0:-neurons] #todas las filas y desde la primera colm hasta la penultima
        data_labels = data_arr[:,-neurons:]#todas las filas y la ultima colum

        print(f"data_features: {data_features}")
        print(f"data_labels: {data_labels}")
        if neurons == 1:
            data_labels = data_labels.reshape(-1, 1)
        
        #make sure the dimentions
        print(f"Dimentions : {data_labels.shape}")
       
        if norm == True:

            #Lets normalize the data
            scaler = StandardScaler() #Create an object of this library in particular

            data_features_norm = scaler.fit_transform(data_features)
            data_labels_norm = scaler.fit_transform(data_labels)
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        #Lets split the data into training an testing
        #-----------------Siempre en este orden ------------------10% en test en y 90% en trainig
        
        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
        print(f'Features: {train_features}')
        print(f'Labels: {train_labels}')
        return train_features, test_features, train_labels, test_labels

