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
    
    def data_process(self,file):
        ##Absolute path:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","data"))
        
        ##Find excel route
        excel_path = os.path.join(data_dir,file)
        
        ##load the excel file
        data_raw = pd.read_excel(excel_path,sheet_name = 0)
        
        data_arr = np.array(data_raw)
        
        #Split in features(inputs) and labels(outputs)
        data_features = data_arr[:,0:-1]
        data_labels = data_arr[:,-1]
        
        ##Check for strings
        str_cols = np.empty(data_arr.shape[1], dtype = bool)
        
        for i in range(0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]),np.str_)
        
        for i in range(0,data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:,i] = le.fit_transform(data_arr[:,i]) + 1
            
            
            
        #Normalize data
        scaler = StandardScaler()
        data_features_norm = scaler.fit_transform(data_features)
        data_labels_norm = scaler.fit_transform(data_labels)
        
        print()

        
        
        