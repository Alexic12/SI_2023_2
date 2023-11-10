import os
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

##lets import libraries
import tensorflow as tf 
from tensorflow import keras 

class Data:
    """
    Attributes:

    Methods:

    """

    def __init__(self):
        self.scaler = StandardScaler() ##Create an object of this library in particular
        self.max_value = 0
        self.array_size = 0
        self.control_vector_size = None

    def data_process(self,file,test_split,norm,neurons,avoid_col):
        ##Let's define the absolute path for this folder 
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##Loads the raw excel file 
        data_raw =  pd.read_excel(excel_path, sheet_name = 0)

        ##Let's store the original features
        columns = data_raw.shape[1]
        original_features = data_raw[data_raw.columns[:columns-neurons]]
        original_labels = data_raw[data_raw.columns[columns-neurons:columns]]
        print(f'Original Features: {original_features}')
        print(f'Original Labels: {original_labels}')

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
        data_features = data_arr[:,avoid_col:-neurons] ## all columns but the last n
        data_labels = data_arr[:,-neurons:] ## The last n columns  

        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)      

        ##Let's check the dimensions of the arrays
        #print(f'Dimensions: {data_labels.shape}')

        ##Let's normalize the data
        if norm == True:
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)
            
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

        return train_features, test_features, train_labels,test_labels,original_features,original_labels

    def download_database(self, database):
        if database == "MNIST":
            (train_images,train_labels),(test_images,test_labels) = keras.datasets.mnist.load_data()

        elif database == "CIFAR10":
            pass

        elif database == "CIFAR100":
            pass

        return train_images,test_images,train_labels,test_labels

    def timeseries_process(self,window_size,horizon_size, file,test_split,norm,identificacion):
        ##Let's define the absolute path for this folder 
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##Loads the raw excel file 
        data_raw =  pd.read_excel(excel_path, sheet_name = 0)

        ##Let's turn it into an array
        array_raw = np.array(data_raw)

        ##Let's get the number of sample times for the whole data (#number of sample times)
        data_length = array_raw.shape[1]

        print(f'Data Length for the time series: {data_length}')

        ##Let's create the data base array for storing the data the proper way
        ##Rows = (#Sample Times - Window - Horizon + 1)
        ##Columns = (window + horizon)
        time_series_array = np.zeros((data_length - window_size - horizon_size + 1,window_size*2 + horizon_size + 1))

        ##we have to look through all the raw data and take the correct data points and store them as window and horizon
        for i in range(data_length - window_size - horizon_size):
            if identificacion == 'directa':
                vector = np.concatenate((array_raw[1,i:i+window_size+horizon_size], array_raw[0,i:i+window_size+horizon_size]))  ##Cambiar 0 y 1 para identficacion inversa ----> Directa se aprende x ###Toma un mindowsize + 1
                time_series_array[i] = vector
            elif identificacion == 'inversa':
                vector = np.concatenate((array_raw[0,i:i+window_size+horizon_size], array_raw[1,i:i+window_size+horizon_size]))  ##Cambiar 0 y 1 para identficacion inversa ---> Inversa se aprende U
                time_series_array[i] = vector


        ##Let's print this time_series_arr_
        print('Time Series')
        print(time_series_array)

        ##Let's store the original features
        columns = time_series_array.shape[1]
        original_features = data_raw[data_raw.columns[:columns-horizon_size]]
        original_labels = data_raw[data_raw.columns[columns-horizon_size:columns]]
        print(f'Original Features: {original_features}')
        print(f'Original Labels: {original_labels}')

        ##Let's separate between features and labels 
        data_features = time_series_array[:,0:-horizon_size]
        data_labels = time_series_array[:,-horizon_size]

        ##Let's take the max value of the data
        self.max_value = max(data_labels)

        ##Let's take the number of features
        self.array_size = data_features.shape[1]

        #Let's normalize the data
        if norm == True:
            sc = StandardScaler()

            if horizon_size == 1:
                data_labels = data_labels.reshape(-1,1)

            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)
            
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
        
        data = np.hstack((train_features,train_labels))
        #print(data)
        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        excel_filename = 'DATA_PLANTA_ORGANIZADA.xlsx'
        df.to_excel(excel_filename, index=False)  
             

        return train_features,test_features,train_labels,test_labels,original_features,original_labels

    def get_max_value(self):
        return self.max_value
    
    def get_array_size(self):
        return self.array_size
    
    def timeseries_process_adapt(self,window_size,horizon_size, file,test_split,norm,identificacion):
        ##Let's define the absolute path for this folder 
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##Loads the raw excel file 
        data_raw =  pd.read_excel(excel_path, sheet_name = 0)

        ##Let's turn it into an array
        array_raw = np.array(data_raw)

        ##Let's get the number of sample times for the whole data (#number of sample times)
        data_length = array_raw.shape[1]

        print(f'Data Length for the time series: {data_length}')

        ##Let's create the data base array for storing the data the proper way
        ##Rows = (#Sample Times - Window - Horizon + 1)
        ##Columns = (window + horizon)
        time_series_array = np.zeros((data_length - window_size - horizon_size + 1,window_size*4 + horizon_size + 3))

        ##we have to look through all the raw data and take the correct data points and store them as window and horizon
        for i in range(data_length - window_size - horizon_size):
            if identificacion == 'directa':
                vector = np.concatenate((array_raw[0,i:i+window_size+horizon_size], array_raw[3,i:i+window_size+horizon_size],array_raw[2,i:i+window_size+horizon_size], array_raw[1,i:i+window_size+horizon_size]))  ##Cambiar 0 y 1 para identficacion inversa
                time_series_array[i] = vector
            elif identificacion == 'inversa':
                ###PID NO TIENE INVERSA"""
                print("PID NO TIENE INVERSA")
                # vector = np.concatenate((array_raw[1,i:i+window_size+horizon_size], array_raw[3,i:i+window_size+horizon_size],array_raw[2,i:i+window_size+horizon_size], array_raw[0,i:i+window_size+horizon_size]))  ##Cambiar 0 y 1 para identficacion inversa
                # time_series_array[i] = vector


        ##Let's print this time_series_arr_
        print('Time Series')
        print(time_series_array)

        ##Let's store the original features
        columns = time_series_array.shape[1]
        original_features = data_raw[data_raw.columns[:columns-horizon_size]]
        original_labels = data_raw[data_raw.columns[columns-horizon_size:columns]]
        print(f'Original Features: {original_features}')
        print(f'Original Labels: {original_labels}')

        ##Let's separate between features and labels 
        data_features = time_series_array[:,0:-horizon_size]
        data_labels = time_series_array[:,-horizon_size]

        ##Let's take the max value of the data
        self.max_value = max(data_labels)

        ##Let's take the number of features
        self.array_size = data_features.shape[1]

        #Let's normalize the data
        if norm == True:
            sc = StandardScaler()

            if horizon_size == 1:
                data_labels = data_labels.reshape(-1,1)

            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)
            
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
        
        data = np.hstack((train_features,train_labels))
        self.control_vector_size = (data.shape[1])- 1
        print(f"NUM IMPUTS: {self.control_vector_size}")

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        excel_filename = 'DATA_PID_ORGANIZADA_ADAPT.xlsx'
        df.to_excel(excel_filename, index=False)  
             

        return train_features,test_features,train_labels,test_labels,original_features,original_labels

    def numInputs(self):
        return self.control_vector_size

            


# ##Let's run this method of time series just for testing
# T = Data()
# T.timeseries_process(3,1,'DATA_SENO_DIRECTO.xlsx')
  
