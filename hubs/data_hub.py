import os
import sys

import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

class Data:
    """
    Attributes:


    Methods:

    """

    def __init__(self) -> None:
        pass

    def data_process(self, file, test_split):
        ##defining the absolute path for data folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the comlplete path for data file route
        excel_path = os.path.join(data_dir,file)

        ##load the raw excel file
        data_raw = pd.read_excel(excel_path , sheet_name=0 )

        ##convert data to array
        data_arr = np.array(data_raw)

        ##let split the data into feature(inputs) and labels(outputs)
        data_features = data_arr[:,0:-1]
        data_labels = data_arr[:,-1]

        data_labels = data_labels.reshape(-1,1)

        ##lets check the dimension of the array
        print(f'Dimension : {data_features.shape}')

        #lets label encode any text in the data

        #lets create a boolean array with the size of the columns of the array
        str_cols = np.empty(data_arr.shape[1],dtype=bool)

        #lets read columns data type
        for i in range(data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(data_arr.shape[1]):
                if(str_cols[i]):
                    le = LabelEncoder()
                    data_arr[:,i] = le.fit_transform(data_arr[:,i]) +1
        #lets normalize the data
        scaler = StandardScaler()#Create an object of this library in particular
        
        data_features_normalize = scaler.fit_transform(data_features)
        data_labels_normalize = scaler.fit_transform(data_labels)

        print(data_labels_normalize)

        #lets split the data into train and test
        #input (train, test) output
        if test_split != 0:
            train_feateures, train_labels, test_labels = tts(data_features_normalize,data_labels_normalize, test_size = 0.1)
        else:
             test_feateures = 0
             test_labels = 0
             train_feateures = data_features_normalize
             train_labels = data_labels_normalize
             

        train_feateures, test_feateures, train_labels, test_labels = tts(data_features_normalize, data_labels_normalize, test_size=0.1)

        return train_feateures, test_feateures, train_labels, test_labels
        






