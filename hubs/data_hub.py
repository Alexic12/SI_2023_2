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


    def data_process(self, file):
        #Obtiene la ruta del directorio data
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

        #Obtiene la ruta del archivo excel
        excel_path = os.path.join(data_dir, file)

        #Carga el archivo excel
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        #Convierte el archivo excel en un array
        data_arr = np.array(data_raw)

        #Label Encode el texto en los datos
        str_cols = np.empty(data_arr.shape[1], dtype=bool) #Crea un array de booleanos del tamaño de las columnas

        for i in range(data_arr.shape[1]): #Recorre las columnas
            str_cols[i] = np.issubdtype(type(data_arr[0, i]), np.str_) #Si el elemento es un string, esa posición es True
        
        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:, i]) + 1

        #Separa los datos en Features (Input) y Labels (Output)
        data_features = data_arr[:, 0:-1] #Todas las filas, todas las columnas menos la última
        data_labels = data_arr[:, -1]     #Todas las filas, solo la última columna

        #Normaliza los datos
        scaler = StandardScaler() #Crea el objeto scaler

        data_features_norm = scaler.fit_transform(data_features) #Normaliza los datos de entrada
        data_labels_norm = scaler.fit_transform(data_labels.reshape(-1, 1)) #Normaliza los datos de salida

        print(data_features_norm)