import os 
import sys

import pandas as pd
import numpy as np

import sklearn.preprocessing 
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts


class Data:
    """
    Attribute:

    Methods: 

    """
    def __init__(self) : 
        pass
    
    def data_process(self, file):
        ## Lets define the absolute path for this folder 
        data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data')) 
        #va a mirar la carpeta data que esta afuera

        ##Find the complete excel file route
        excel_path = os.path.join(data_dir, file)

        ##LEts load the raw excel file 
        data_raw = pd.read_excel(excel_path, sheet_name=0)

        ##Lets confert the raw data to an array
        data_arr = np.array(data_raw)

        ##Lets split the data into features and labels 
        data_features = data_arr[: , 0:-1]
        data_labels = data_arr[: , -1]

        ##
