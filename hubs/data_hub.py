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

    def data_process (self, file,test_split):
        #Definir la ruta principal de esta carpeta
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        #Ruta del archivo de excel
        excel_path = os.path.join (data_dir, file)

        #Leyendo el archivo de excel
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        #Convertir el archivo en un array
        data_arr = np.array(data_raw)

        #Crear una fila de boleanos del mismo tamaño del arreglo
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        #Leer el tipo de columnas
        for i in range(0,data_arr.shape[1]):
            #Vamos detectar si una columna tiene un string o un numero por medio de un booleano
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            #Mira si la columna booleana que creamos en int o String
            if str_cols[i]:
                le = LabelEncoder()
                #Transforma todos los strings en int
                data_arr[:,i] = le.fit_transform(data_arr[:,i])+1


        #Diferenciar los features de los labels
        data_features = data_arr[:,0:-1]
        data_labels = data_arr[:,-1]

        data_labels= data_labels.reshape(-1,1)

        
        #print(f'Dimensions:{data_labels.shape}')

        #Normalizar los datos 
        scaler = StandardScaler() #Crear un objeto de esta libreria en particular

        data_features_norm = scaler.fit_transform(data_features)#Normalizando las featurs (input)
        data_labels_nomr = scaler.fit_transform(data_labels) #Nomalizar los labels (output)

        #dividir los datos de entrenamiento con los de testeo
        #mantener el mismo orden entrada->(entreno, testeo) salida(entreno, testeo)
        if test_split!=0:
            train_features, test_features, train_leables, test_leables = tts(data_features_norm, data_labels_nomr, test_size=test_split)
        else:
            test_features = 0
            test_leables = 0
            train_features = data_features_norm
            train_leables = data_labels_nomr

        
        return train_features, test_features, train_leables, test_leables
    


