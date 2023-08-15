import os # to search
import sys # to use drivers
import pandas as pd
import numpy as np

import sklearn.preprocessing # to process information to normalize data 
import sklearn.model_selection # subdivision to training,
from   sklearn.preprocessing import StandardScaler 
from   sklearn.preprocessing import LabelEncoder
from   sklearn.model_selection import train_test_split as tts # just for data processing

class Data: # docstring what are input/output & what is doing the class? 
    """

    Attributes:
    Methods:

    """
    def _init_(self):
        pass

    def data_process(sef, file, test_split, norm, neurons):
        #lets difine the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data')) # entra al folder y tomas el datos dentro del folder 
        excel_path = os.path.join(data_dir, file) # donde esta el archivo de excel # Find the complete excel file route
        # ahora vamos a abrir y procesar 
        #lets load the raw excel file 
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        # lets confert the raw data to an array
        data_arr = np.array(data_raw)

        #aqui vamos a revisar si es un texto o un dato 
        #lets create a boolean array with the size of the colum
        str_cols = np.empty(data_arr.shape[1],dtype=bool) # creamos un array vacio con el tamana el la info y en tipo booleano
        ## lets read columns data type 
        for i in range(0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]),np.str_) # 0 y i # aqui revisamos. 
        for i in range (0,data_arr.shape[1]):
            if str_cols[i]:
                le= LabelEncoder()
                data_arr[:,i]= le.fit_transform(data_arr[:,i])+1 # ahora tenemos todos los datos en en tipo texto 
        #lets split the data into features ans labels 
        data_features= data_arr[:,0:-neurons] #todas las columnas menos la ultima. [filas,columnas]
        data_labels= data_arr[:,-neurons:]    # sola la ultima columna

        print(f'DATALABELS:{data_labels}')
        print(f'DATALABELS:{data_features}')
        if neurons == 1:
            data_labels = data_labels.reshape(-1,1)

        #Lets label encode any text in the data( debe ser matematico operation) vamos a revisar los datos y vamos a revisar que es un string y int  cuales tiene text , revirara y poner true or false

        # lets check the dimensions of the arrays 
        #print(f'Dimensions:{data_labels.shape}')

        # vamos a noramalizar los datos con esta libreria 
        #lets normalize the data 
        if norm == True:
            scaler = StandardScaler()##Create an object solo es para esta liberia # create an object if this library in particular 

            data_features_norm = scaler.fit_transform(data_features)
            data_labels_norm = scaler.fit_transform(data_labels)
            # Ahora tenemos el data normalizado el cacula solo y busca el mejor intervalo
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels


        #print(data_labels_norm)

        # vamos a crear un condicional
        if test_split !=0:
          Train_features, test_features, train_label, test_labels = tts(data_features_norm,data_labels_norm, test_size=test_split)
        else:
            test_features=0
            test_labels = 0
            Train_features=data_features_norm
            train_label=data_labels_norm
        print(f'Feature: {Train_features}')
        print(f'Labels: {train_label}')

        return train_label,test_features,train_label,test_labels
    

