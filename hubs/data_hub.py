import os #buscar carpetas en el sistema
import sys #usar todos los drivers del sistema

import pandas as pd #administracion de datos 
import numpy as np 

import sklearn.preprocessing
import sklearn.model_selection

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts  #unicamente para procesamiento de datos
from sklearn.preprocessing import LabelEncoder

class Data:
    """
    Atributes:

    methods: 
    """

    def __init__(self):
        pass 
    
    def data_process(self,file): #recibe una ruta 
        ## Lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.ditname(__file__),'..','data'))

        excel_path = os.path.join(data_dir,file) #Aqui se pone la dirección del documento como tal

        #lets load the raw excel file 
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        #convert the data to an arraw to work with it
        data_arr = np.array(data_raw)

        # lets label encode any text in the data
        str_cols = np.empty(data_arr.shape[1],dtype=bool)

        ## lets read colums data type
        for i in range (0,data_arr.shape[1]): ## detecta si es un string para marcar las columnas que corresponden como verdadero
            str_cols[i] = np.issubdtype(type(data_arr[0],i),np.str_)

        for i in range(0,data_arr.shape[1]): ## detecta si es un booleano para marcar las columnas que corresponden como verdadero
            if str_cols[i]:
                le=LabelEncoder()
                data_arr[:,i]=le.fit_transform(data_arr[:,i])+1

        #lets split the data into features(inputs) and labels(outputs)
        data_features = data_arr[:,0:-1]  
        data_labels = data_arr[:,-1]

        #lets normalize the data
        scaler= StandardScaler() ##create an object of this library in particular 

        data_features_norm = scaler.fit_transform(data_features)
        data_labels_norm = scaler.fit_transform(data_labels)

        print(data_labels_norm)









