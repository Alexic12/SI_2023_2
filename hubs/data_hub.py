import os
import sys
import pandas as pd
import numpy as np 
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts 
import tensorflow as tf 
from tensorflow import keras


class data:
    """
    attributes:
    
    methods:
    
    """
    
    def __init__(self):
        
        self.scaler = StandardScaler()##create an object of this library 

    
    def data_process(self,file,test_split,norm,neurons,avoid_col):
        ##lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
        
        ##Find the complete excel file route 
        excel_path = os.path.join(data_dir,file)
        
        ##lets load the raw excel file 
        data_raw = pd.read_excel(excel_path,sheet_name=0)
        
        ##lets store the original feature
        
        columns = data_raw.shape[1]
        original_feature = data_raw[data_raw.columns[:columns-neurons]]
        original_labels = data_raw[data_raw.columns[columns-neurons:columns]]
        print(f'Original_features {original_feature}')
        print(f'Original_labels{original_labels}')
        
        
        ##lets confert the raw data to an array 
        data_arr = np.array(data_raw)
        
        ##lets label encode any text in the data
        
        ##lets create a boolean array with the size of the columns  
        str_cols = np.empty(data_arr.shape[1], dtype=bool)
        
        ## lets read columns data type
        for i in range(0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)
            
        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:,i] = le.fit_transform(data_arr[:,i]) + 1
        
        ##lets split the data into features(inputs) and labels(outputs)
        data_features = data_arr[:,avoid_col:-neurons]
        data_labels = data_arr[:,-neurons:]
        
        print(f'data_features: {data_features}')
        print(f'data_labels: {data_labels}')
        
        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)
        
        ##lets check the dimensions of the arrays 
        
        ##print(f'Dimension:{data_labels.shape}')
        
        if norm == True:
            ## lets normalize the data 
        
            data_features_norm = self.scaler.fit_transform(data_features) 
            data_labels_norm = self.scaler.fit_transform(data_labels)
            
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels
        
        ##print(data_labels_norm)
        
        ##lets split the data into training and testing 
        ##input (train,test) output(train, test)
        
        if test_split != 0:
        
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split)
            
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'features: {train_features}')
            print(f'labels: {train_labels}')
        
        
        
        return train_features, test_features, train_labels, test_labels, original_feature, original_labels
    
    
    def denormalize(self, data):
        
        data_denorm = self.scaler.inverse_transform(data)
        
        return data_denorm
    
    
    def download_database(self, database):
        if database == 'MNIST':
            
            (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
            
            
        elif database == 'CIFAR10':
            
            pass
            
        elif database == 'CIFAR100':
            
            pass
        
        return train_images, test_images, train_labels, test_labels
