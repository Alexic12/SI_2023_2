import os 
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing 
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts


class Data:
    """
    Attribute:

    Methods: 

    """
    def __init__(self) : 
        pass
    
    def data_process(self, file, test_split, norm, neurons, avoid_col):
        ## Lets define the absolute path for this folder 
        data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data')) 
        #va a mirar la carpeta data que esta afuera

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##LEts load the raw excel file 
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        ##Lets confert the raw data to an array
        data_arr = np.array(data_raw)

        ##Lets label encode any text in the data

        ##Lets create a boolean array with the size of the columns
        str_cols = np.empty(data_arr.shape[1], dtype = bool)

        ##Lets read columns data type
        for i in range (0, data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)
        
        for i in range (0 , data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[: , i] = le.fit_transform(data_arr[:, i]) + 1

        ##Lets split the data into features and labels 
        data_features = data_arr[: , avoid_col :-neurons]
        data_labels = data_arr[: , -neurons:]

        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)
        #lets check the dimensions of the array 
        # print(f'Dimensions:{data_labels.shape}')

        if norm == True:
                

            ##Lets normalize the data
            scaler = StandardScaler()##Create an object of this library in particular
            data_features_norm = scaler.fit_transform(data_features)
            data_labels_norm = scaler.fit_transform(data_labels)

            # print(data_labels_norm)

        else:
            data_features_norm = data_features
            data_labels_norm = data_labels
        #Lets split the data into training and testing
        ##input (train, test) output (train, test)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
        
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'Features: {train_labels}')


        

        return train_features, test_features, train_labels, test_labels 