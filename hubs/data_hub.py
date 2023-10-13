import os
import sys #podemos usar el GPU del computador

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

import tensorflow
from tensorflow import keras

class Data:
    """
    
    """
    def __init__(self):
        self.scaler = StandardScaler()##Create an object of this library in particular

    def data_process(self, file, test_split, norm, neurons, avoid_col):
        ##lets define the absolute path for this folder
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) # __file__ es un atributo
                                                                                    #que me permite obtener el
                                                                                    #path del archivo .py donde
                                                                                    #lo uso, en este caso, 
                                                                                    # obtengo el path de data_hub.py

                                                                                    # os.path.dirname devuelve todo 
                                                                                    # el path sin la parte del
                                                                                    # archivo, es decir, todo el 
                                                                                    # directorio

                                                                                    # os.path.join une los path
                                                                                    # que pongamos, literal solo 
                                                                                    # los concatena poniendo /
                                                                                    # donde se necesite, ponemos 
                                                                                    # '..' para subir un nivel
                                                                                    # en las carpetas, entonces
                                                                                    # el computador llega a hubs
                                                                                    # ve que hay '..' y entonces 
                                                                                    # sube una jerarquia, luego 
                                                                                    # de subir sigue con el 
                                                                                    # resto del path

                                                                                    # os.path.abspath devuelve el 
                                                                                    # path completo, en esta 
                                                                                    #parte es que el sistema usa
                                                                                    # los '..' y llega al path
                                                                                    # que queremos
        ##find the complete excel file route
        excel_path = os.path.join(data_dir, file)
        ##load he raw excel file
        data_raw = pd.read_excel(excel_path, sheet_name = 0) # sheet name en cero nos da la primera pagina, podemos
                                                            #poner el nombre entre comillas de la pagina si queremos
                                                            # Por cierto hay que tener en cuenta que este no 
                                                            # pone en la fila 0 a los titulos de las
                                                            # columnas, empieza a numerar desde la primera fila 
                                                            # que tenga numeros

        ##lets store the original features
        columns = data_raw.shape[1]
        original_features = data_raw[data_raw.columns[:columns-neurons]]
        original_labels = data_raw[data_raw.columns[columns-neurons:columns]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        ##lets convert the raw data to an array
        data_arr = np.array(data_raw)

        ##lets label encode any text in the data
        ##first create a boolean array with the size of the column
        str_cols = np.empty(data_arr.shape[1], dtype=bool) # creamos un arreglo de tipo booleano no inicializado
                                                        # que significa que pone valores al azar, deben ser llenados
                                                        # despues por el usuario, entonces le decimos que sea 
                                                        # de la cantidad de columnas de data_arr y que sea de tipo 
                                                        # booleano pero que no inicialice las variables, es decir,
                                                        # son al azar 

        ##lets read columns data type data_arr.shape[1] es la cantidad de columnas
        for i in range(0, data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0, i]), np.str_) # la fila cero no son los titulos de las
                                                                    # columnas, desde la cero ya hay datos, estamos
                                                                    # buscando cuales son str y cuales son int

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder() 
                data_arr[:, i] = le.fit_transform(data_arr[:, i]) + 1 #se suma uno porque empieza en cero, el 
                                                                # label encoder lo que hace es convertir valores
                                                                # categoricos en valores numericos, se supone que 
                                                                # los convierte de forma logica en un orden 
                                                                # determinado desde cero, por esto se le suma 1,
                                                                # para que no empiece desde cero
        ##lets split the data into features and labels
        data_features = data_arr[:,avoid_col:-neurons] # la ultima columna no en este caso
        data_labels = data_arr[:, -neurons:]

        if neurons == 1:
            data_labels = data_labels.reshape(-1, 1) #necesitabamos que fuera de dos dimensiones por lo menos
                                                # basicamente el -1 en el reshape dice al sistema que no 
                                                # sabemos de cuanto va a quedar esa dimension, que el sistema
                                                # debe darse cuenta solo, segun la cantidad de datos del arreglo
                                                # que estamos reformando, en este caso data_labels.reshape(-1, 1)
                                                # significa que no sabemos cuantas filas va a haber, pero si que
                                                # queremos que halla exactamente una columna, ya que pusimos 1 en
                                                # la parte de la columna

###############################################################################################
# por que el data_features se actualizo al actualizar data_arr, es decir, por que cuando le hice el fit_transform
# a data_arr esto tambien modifico a data_features y porque no queremos que empiece desde cero con el labelencoder
###################################################################################################
        ##lets normalize the data
        if norm == True:
            data_features_norm = self.scaler.fit_transform(data_features)
            data_labels_norm = self.scaler.fit_transform(data_labels)  

        else:
            data_features_norm = data_features
            data_labels_norm = data_labels
###############################################################################################
# por que al hacer el scaler.fit_transform algunos datos no quedaron entre -1 y 1
###################################################################################################

        # lets split the data into training and testing

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split) 
        #nos da los elementos en el orden que los escribimos al hacer las variables
        # test_size=0.1 es tomar el 10% de la data al azar para el test y entonces va a usar el 90% para entrenar
        # literalmente tts nos toma al azar en el dataset para el entrenamiento y para el testeo
        # los datos se ponen como train_x, test_x, train_y, test_y

        else: 
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'Labels: {train_labels}')

        return train_features, test_features, train_labels, test_labels, original_features, original_labels

    def download_database(self, database):
        if database == 'MNIST':
            (train_images, train_labels),(test_images, test_labels) = keras.datasets.mnist.load_data()

        elif database == 'CIFAR10':
            pass
        elif database == 'CIFAR100':
            pass

        return train_images, test_images, train_labels, test_labels 

    def timeseries_process(self, window_size, horizon_size, file, test_split, norm):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        excel_path = os.path.join(data_dir, file)
        data_raw = pd.read_excel(excel_path, sheet_name = 0)

        array_raw = np.array(data_raw)

        data_length = array_raw.shape[1]

        print(f'Sample times for time series: {data_length}')

        #lets create the data base array for storing the data in a proper way
        print(type(window_size*2 + horizon_size + 1))
        time_series_arr = np.zeros((data_length - window_size - horizon_size + 1, window_size*2 + horizon_size + 1))

        for i in range(data_length - window_size - horizon_size):
            vector = np.concatenate((array_raw[0, i:i+window_size+horizon_size], array_raw[1, i:i+window_size+horizon_size]))
            time_series_arr[i] = vector

        print('time series')
        print(time_series_arr)

        ##lets store the original features
        columns = time_series_arr.shape[1]
        original_features = data_raw[data_raw.columns[0:-horizon_size]]
        original_labels = data_raw[data_raw.columns[-horizon_size:]]
        print(f'Original_features {original_features}')
        print(f'Original_labels {original_labels}')

        data_features = time_series_arr[:, 0:-horizon_size]
        data_labels = time_series_arr[:, -horizon_size:]

        if norm:
            if horizon_size == 1:
                data_labels = data_labels.reshape(-1, 1)
            sc = StandardScaler()
            data_features_norm = sc.fit_transform(data_features)
            data_labels_norm = sc.fit_transform(data_labels)
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels


        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm, test_size=test_split) 

        else: 
            test_features = 0
            test_labels = 0
            train_features = data_features_norm
            train_labels = data_labels_norm
            print(f'Features: {train_features}')
            print(f'Labels: {train_labels}')
        
        return train_features, test_features, train_labels, test_labels, original_features, original_labels

#T = Data()
#T.timeseries_process(3, 1, 'DATA_SENO_DIRECTO.xlsx')