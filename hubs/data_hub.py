import os #For finding routes
import sys # Import and use drivers for the system

import pandas as pd #Data management, array spliting, array creation.
import numpy as np # Numerical python.

import sklearn.preprocessing #Normalize data at least [0 -1] or  [-1 -1]
import sklearn.model_selection #Splitting the data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts

class Data:
    """
    Attributes:


    Methods:
    #
    """
    #self is for OOP

    def __init__(self):
        pass

    def data_process(self, file):
        ##Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file) #Completes the route

        ##Lets load the raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        ##Convert data into an array to work with it
        data_arr = np.array(data_raw)

        ##Lets label encode any text in the data
        #boolean array with the size of columns of the original data array
        str_cols = np.empty(data_arr.shape[1], dtype=bool) 

        #Lets read columns data type
        for i in range(0, data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                ##Will be a string
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:,i]) + 1

        ##Lets split the data into features and labels
        data_features = data_arr[:,0:-1]
        data_labels = data_arr[:, -1]

        data_labels = data_labels.reshape(-1,1) #Adding one plus dimension

        ##Checking the array dimensions

        #print(f'Dimensions: {data_labels.shape}')

        ##Note: scaler.fit... uses at least 2 dimensions

        ##Lets normalize the data
        # It decides by itself scalation interval
        scaler = StandardScaler() ## Create an object of this library in particular
        data_features_norm = scaler.fit_transform(data_features)
        data_labels_norm = scaler.fit_transform(data_labels)

        #print(data_labels_norm)

        # train-test split
        #returns 4 elements in this order, features are input data labels are output data
        ##input  (train,test) output (train, test)
        train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=0.1)

        return train_features, test_features, train_labels, test_labels