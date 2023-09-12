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

    def data_process(self, file, test_split, norm, neurons, avoid_column):
        # Obtiene la ruta del directorio data
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data")
        )

        # Obtiene la ruta del archivo excel
        excel_path = os.path.join(data_dir, file)

        # Carga el archivo excel
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        # Convierte el archivo excel en un array
        data_arr = np.array(data_raw)

        # Label Encode el texto en los datos
        str_cols = np.empty(
            data_arr.shape[1], dtype=bool
        )  # Crea un array de booleanos del tamaño de las columnas

        for i in range(data_arr.shape[1]):  # Recorre las columnas
            str_cols[i] = np.issubdtype(
                type(data_arr[0, i]), np.str_
            )  # Si el elemento es un string, esa posición es True

        for i in range(0, data_arr.shape[1]):
            if str_cols[i]:
                le = LabelEncoder()
                data_arr[:, i] = le.fit_transform(data_arr[:, i]) + 1

        # Separa los datos en Features (Input) y Labels (Output)
        data_features = data_arr[:, avoid_column:-neurons]  # Desde la columna descrita, menos las últimas columnas
        data_labels = data_arr[:, -neurons:]  # Todas las filas, solo las últimas columnas (neuronas)

        #print(f'data labels: {data_labels} \n data features: {data_features}')

        if neurons == 1:
            data_labels = data_labels.reshape(-1, 1)  # Cambia la forma de la matriz de (n,) a (n, 1)

        # Revisar dimensiones de los datos
        # print(f'Dimensiones: {data_labels.shape}')

        if norm:
            # Normaliza los datos
            scaler = StandardScaler()  # Crea el objeto scaler

            data_features_norm = scaler.fit_transform(
                data_features
            )  # Normaliza los datos de entrada
            data_labels_norm = scaler.fit_transform(
                data_labels
            )  # Normaliza los datos de salida
        else:
            data_features_norm = data_features
            data_labels_norm = data_labels

        # print(data_features_norm)

        if test_split != 0:
            train_features, test_features, train_labels, test_labels = tts(
                data_features_norm, data_labels_norm, test_size=test_split
            )
        else:
            test_labels = 0
            test_features = 0
            train_features = data_features_norm
            train_labels = data_labels_norm

        return train_features, test_features, train_labels, test_labels
