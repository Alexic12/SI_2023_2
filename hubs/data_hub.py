import os #Sistema operativo
import sys #Sistema operativo

import pandas as pd #base de datos
import numpy as np #operaciones numericas

import sklearn.preprocessing #biblioteca de aprendizaje automatico
import sklearn.model_selection

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

class Data:
    #Hace el procesamiento y organizacion de datos que recibe desde el Neural hub
    """
    Attributes:

    Methods:
    """
    def _init_(self):
        pass

    def data_process (self, file,test_split,norm,neurons): # definimos la funcion de procesamiento de datos
        
        #Definir la ruta principal de esta carpeta
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))

        #Ruta del archivo de excel
        excel_path = os.path.join (data_dir, file)

        #Leyendo el archivo de excel
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        #Convertir el archivo en un array
        data_arr = np.array(data_raw)
        print(data_arr)

        #Crea fila de booleanos del mismo tamaño del arreglo
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        #Leer y recorrer las columnas
        for i in range(0,data_arr.shape[1]):
            #Vamos detectar si una columna tiene un string o un numero por medio de un booleano
            str_cols[i] = np.issubdtype(type(data_arr[0,i]), np.str_)

        for i in range(0, data_arr.shape[1]):
            #Mira si la columna booleana que creamos es int o String
            if str_cols[i]:
                le = LabelEncoder() #la funcion LabelEncoder transforma todos los strings en numeros int

                data_arr[:,i] = le.fit_transform(data_arr[:,i])+1


        #Diferenciar los features de los labels
        data_features = data_arr[:,0:-neurons]  #datos, son todas las filas y las columnas desde la 0 a 5
        data_labels = data_arr[:,-neurons:-1]   #resultado, son todas las filas y la columna 6 

        print (f'DATA_FEATURES: {data_features}')
        print (f'DATA_LABELS: {data_labels}')

        if neurons == 1:
            data_labels= data_labels.reshape(-1,1) # Cambia la forma de los labels para que sean bidimensionales y se puedn usar (por un parametro de scikit_learn)

        
        # print(f'Dimensions:{data_labels.shape}')

        if norm == True:
            #Normalizar los datos 
            scaler = StandardScaler() #Crear un objeto de esta libreria en particular
            data_features_norm = scaler.fit_transform(data_features)#Normalizando las featurs (input)
            data_labels_norm = scaler.fit_transform(data_labels) #Nomalizar los labels (output)
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels
        
        if test_split != 0:
            #Dividir los datos entre los training y testing
            #input (train,test) output(train,test)
            train_features,test_features,train_labels,test_labels=tts(data_features_norm,data_labels_norm,test_size=0.1)
        else:
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'features::{train_features}')
            print(f'labels::{train_labels}')

        return train_features, test_features, train_labels, test_labels