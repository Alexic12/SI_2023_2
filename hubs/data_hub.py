import os

import sys

 

import pandas as pd

import numpy as np

 

import sklearn

 

import sklearn.preprocessing

import sklearn.model_selection #separa la data en entreno y luego en test para ir enseñandole al computador y verificar que si coja los datosbn

from sklearn.preprocessing import StandardScaler#normaliza la data

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split as tts

 

class Data:

    """""

    Attributes:

 

    Methods:

 

    """""

 

def __init__(self):

    pass

 

def data_process (self,file):

    ##Lets define the absolute pat for this folder

    data_dir= os.path.abspath(os.path.join(os.path.dirname(__file__),"..","data"))

 

    ##Find the complete excel file route

    excel_path = os.path.join(data_dir,file)

 

    ##lets load the raw excel file

    data_raw= pd.read_excel(excel_path, sheet__name = 0)

 

    ##Lets convert the raw data to an array

    data_arr= np.array(data_raw)

 

    ##Lets label encode any text in the data

 

 

    ##Lets create a boolean array with the size of de colum

    str_cols = np.empty(data_arr.shape[1], dtype= bool)

 

    ##Lets read columns data type

    for i in range(0,data_arr.shape[1]):

        str_cols[i] = np.issubdtype(type(data_arr[0,i],np.str_))

   

    for i in range(0,data_arr.shape[1]):

        if str_cols[i]:

            le = LabelEncoder()

            data_arr[:,i] = le.fit_transform(data_arr[:,i]) + 1

 

    ##Lets split the data into features and labels

    data_features = data_arr[:,0:-1]##la ultima columna es lo que queremos que aprenda ent se toman todas lascolumnas menos la ultima

    data_labels = data_arr[:,-1]

   

    ##Lets normalize the data

    scaler = StandardScaler()##Creante an object of this library in particular

 

    data_features_norm = scaler.fit_transform(data_features)

    data_labels_norm = scaler.fit_transform(data_labels)

 

    print(data_labels_norm)