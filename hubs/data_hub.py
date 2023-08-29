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

    def data_process(self,file,test_split,norm,neurons):
        ##Let's define the absolute path for this folder 
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##Loads the raw excel file 
        data_raw =  pd.read_excel(excel_path, sheet_name = 0)

        ##Let's convert the raw data to an array
        data_arr = np.array(data_raw)
        print(f'Data: {data_arr}')

        ##Let's label encode any text in the data

        #Let's create a boolean array with size of the columns
        str_cols = np.empty(data_arr.shape[1], dtype = bool)

        #Let's read columns data type
        for i in range (0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range (0,data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:,i] = le.fit_transform(data_arr[:,i]) + 1

        ##Let's split the data into features (inputs) and labels(outputs)
        data_features = data_arr[:,0:-neurons] ## all columns but the last n
        data_labels = data_arr[:,-neurons:] ## The last n columns  

        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)      

        ##Let's check the dimensions of the arrays
        #print(f'Dimensions: {data_labels.shape}')

        ##Let's normalize the data
        if norm == True:
            scaler = StandardScaler() ##Create an object of this library in particular

            data_features_norm = scaler.fit_transform(data_features)
            data_labels_norm = scaler.fit_transform(data_labels)
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        ##Let's split the data into training and testing
        ##input(train,test) output(train,test)
        if test_split != 0:
            train_features, test_features, train_labels,test_labels = tts(data_features_norm,data_labels_norm, test_size = test_split)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'Labels: {train_labels}')

        return train_features, test_features, train_labels,test_labels

  
