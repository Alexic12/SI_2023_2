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
    Atributes:
    
    Methods:
    
    """
    def __init__(self):
        pass
    
    def data_process(self,file):
        #los .. es para subir un nivel en el directorio y luego se va a la carpeta data
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
        excel_path = os.path.join(data_dir,file)
        
        data_raw = pd.read_excel(excel_path, sheet_name = 0)
        data_array = np.array(data_raw)
        
        #lets encode any string in the data
        str_cols = np.empty(data_array.shape[1], dtype=bool)
        for i in range(0,data_array.shape[1]):
            str_cols[i] = np.issubdtype(type(data_array[0,i]),np.str_)
            if str_cols[i]:
                le = LabelEncoder()
                data_array[:,i] = le.fit_transform(data_array[:,i]) + 1
        
        #lets split the data into features and labels
        data_features = data_array[:,0:-1]
        data_labels = data_array[:,-1]
        
        #lets normalize
        scaler = StandardScaler()#object of the library
        data_features_norm = scaler.fit.transform(data_features)
        data_labels_norm = scaler.fit.transform(data_labels)
        
"""         
        train_features = 
        train_labels = 
        test_features = 
        test_labels =  """