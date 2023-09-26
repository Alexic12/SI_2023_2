#lets import the basic libraries

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# lets import the neural model libraries
import xgboost as xg # metricas lo vamos hacer aparte 

class xgb:
    def __init__(self, depth):
        self.depth = depth ##depth of decision tree

#todas las estructuras o clases de una misma categoria de parametros que entre 
    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        
        ##lets build the model 
        #number of inputs for example(13 inputs, i have a deth of 10 n_estimators will be (inputs+1)*(depth))
        model = self.build_model((train_features.shape[1]+1)*self.depth, alfa, 1) # que profundida va a tener 

        ##lets create an evaluation set.
        eval_set = [(train_features, train_labels),(test_features, test_labels)] # es el set de comprobacion y decir que ifno es entrenamiento

        #lets train the model

        model.fit(train_features, train_labels, eval_metric='mae', eval_set=eval_set, verbose=True) # para poder ver la info y graficar 

        ##lets plot results 

        history = model.evals.result()

        print(history) # como ver las columnas y la informacion para ver como plotear una tabla con el error cuadratico medio.

    
        # vamos hacer un metodo build mode 
        # estos es propio de xgb muy difernete con el ffm_tf es interno de la clase
        # cantidad de capas es la profundidad # verbosity nosmuestra  informacio es le menos importante. cantidad estimadores por profundiadad
    def build_model(self,n_estimators,learning_rate,verbosity): # poder eleguir el clasificador
        model = xg.XGBRegressor(
            # nos va a permitir optimizar si e sun regresor es para series temporales  aproximidar# si usamos clasificador para poder casificar la informacion
            objective='reg:swuarederror',                  # funcion desempeno    # loss function for training determination      
            colsample_byetree = 0.5,                       ##  el 50 se vuelve aletorio  proporcional features that are randomly sampled each iterati(reducing this parameter  can prevent overfitting)(between 0-1)
            learning_rate=learning_rate,                           # la tasa de aprendice porcentaje de error y altera las hojas y las condiones 
            n_estimators=n_estimators, ## numbers of ramifications or branches (neurons)
            reg_lambda=2, #esto no lo tiene los ffm,este amedia que entrene el corta el camino para que se cambie por ramas , para que la informacion que va , la cantidad de cortes que va hacer en cada interacion , dos o 3 cortes. ojo no cortar mas hacen forzar que todas las neuronas se ejerciten 
            gamma=0,# cuando es cero no se le adiciona regulaizar , forzar el lambda aumentar el gamma para quitar lo random 
            max_depth=self.depth, # number of layers
            verbosity=verbosity, ##shows debug info in terminal 
            subsample=0.8, #Esto es par partir la informacion cambia esos datos que tomo. randomply data for trainign for each iteration 0-1
            seed=20,  # sirve para replicar el mismo modelo para los valores aleatorios.los pesos son los mismo iniciales
            tree_method='hist', #  eficiente para bases grandes de datos. comprimir informacion , reduce los calculos
            updater= 'grow_quantile_histmaker,prune'# entrena las ramas , hace los calculos con histograma.  entrenas las hojas,
        # todo el tiempo esta evitadno el overthing.
    ) 
        return model
    

        



    
