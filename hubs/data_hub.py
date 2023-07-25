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
    Attributes:


    Methods:
    
    """
    def __init__(self):
        pass

    def data_process(self, file):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

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
        data_features = data_arr[:,0:-1]
        data_labels = data_arr[:, -1]

        data_labels= data_labels.reshape(-1,1)

        ##Lets normalize the data
        scaler = StandardScaler()##Create an object of this library in particular

        data_features_norm = scaler.fit_transform(data_features)
        data_labels_norm = scaler.fit_transform(data_labels)

        print(data_labels_norm)
        #print(data_labels_norm)

        ## lets split the data into training and testing
        #train_features, test_features, train_labels, test_labels = tts(data_features_norm,data_labels_norm, test_size = 0.1)

        #return train_features, test_features, train_labels, test_labels