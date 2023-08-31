import os
import sys

import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing 
import sklearn.model_selection #separa la data en entreno y luego en test para ir enseñandole al computador y verificar que si coja los datosbn
from sklearn.preprocessing import StandardScaler#normaliza la data
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
 

class Data:

    '''
    Attributes:


    Methods:

    '''

    def __init__(self):

        pass

    def data_process (self,file, test_split, norm, neurons,avoid_col):

        ##Lets define the absolute pat for this folder
        data_dir= os.path.abspath(os.path.join(os.path.dirname(__file__),"..","data"))

    
        ##Find the complete excel file route
        excel_path = os.path.join(data_dir,file)

    
        ##lets load the raw excel file
        data_raw= pd.read_excel(excel_path, sheet_name = 0)

    
        ##Lets convert the raw data to an array
        data_arr= np.array(data_raw)

    ##before split lets label encode any text in the data

        ##Lets create a boolean array with the size of the color
        str_cols = np.empty(data_arr.shape[1], dtype=bool)

        #lets read colums data type
        for i in range (0,data_arr.shape[1]):
            str_cols[i] = np.issubdtype(type(data_arr[0,i]),np.str_)
        
        for i in range(0,data_arr.shape[1]):
            if str_cols[i]:
                le=LabelEncoder()
                data_arr[:,i] = le.fit_transform(data_arr[:,i]) + 1


        ##Lets split the data into features and labels
        data_features = data_arr[:,avoid_col:-neurons]##la ultima columna es lo que queremos que aprenda ent se toman todas lascolumnas menos la ultima
        data_labels = data_arr[:,-neurons:]

        if neurons ==1:
            data_labels= data_labels.reshape(-1,1)

        ##lets  check the dimensions of the arrays
        #print(f'Dimensions : {data_features.shape}')

        ##Lets normalize the data
        if norm==True:

            scaler =StandardScaler()#create an object of this library in particular 
            data_features_norm= scaler.fit_transform(data_features)
            data_labels_norm= scaler.fit_transform(data_labels)
        else:
            data_features_norm= data_features
            data_labels_norm= data_labels
        
        #print(data_labels_norm)

        ##Lets split the data into training and testing
        #input(train, test) output (train,test)
        
        #tts =
        #test size coge el 10% de la dataS

        if test_split !=0:
            train_features, test_features, train_labels, test_labels = tts(data_features_norm, data_labels_norm,test_size = 0.1)
        else: 
            test_features=0
            test_labels=0
            train_features=data_features_norm
            train_labels=data_labels_norm
            print(f'Features: {train_features}')
            print(f'Labels: {train_labels}')
        return train_features, test_features, train_labels, test_labels