##lets import the basic libraries#
# toca correguir este codigo
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##lets import the neural model libraries
import xgboost as xg

## import scikit-learn to the metricks
from sklearn.metrics import accuracy_score as acs

##for denormalizing the data
from sklearn.preprocessing import StandardScaler

class xgb:
    def __init__(self, depth):
        self.depth = depth ##depth of decision tree


    def run(self, train_features, test_features, train_labels, test_labels, original_features, original_labels, iter, alfa, stop_condition,chk_name,train,neurons):
        ##lets build the model
        ##number of inputs  (for example 13 inputs, i have a depth of 10 n_estimators will be (inputs+1)*depth)
        model = self.build_model((train_features.shape[1]+1)*self.depth, alfa, 1) # SI quiero cargar un mdoelo debe ser el mismo configuracion , als mimsa dimenciones.

        ##lets create an evaluation set
        eval_set = [(train_features, train_labels),(test_features, test_labels)]

        if train:
            ##lets train the model
            model.fit(train_features, train_labels, eval_metric='mae', eval_set=eval_set, verbose=True)

            ##lets plot results
            history = model.evals_result()

            ##print(history)
            train_hist = history['validation_0']['mae'] # del vector validation cero vamos a obtener el campo mae 

            plt.figure()
            plt.plot(train_hist, 'r', label='Training Loss Function') # para plotear el history de mae del error.
            plt.xlabel('Epoch')   # poner los labels de x que es son las iteraciones 
            plt.ylabel('mae')     # poner el valor de mae
            plt.title('Training History') # la funcion de desempeno 
            plt.legend()
            plt.show()

            ##validation step
            pred_out = model.predict(test_features)

            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(test_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title('Validation')
            plt.legend()
            plt.show()


            ## lets show the accuracy value fot the this traninng batch

            accuracy = acs(test_labels.astype(int),pred_out.astype(int)) *100 # predicion y el test comparelo que salio y lo que deberia toca convetir en int

            print(f'Accuracy:{accuracy:2f}%') # vamos a ver el entrenamiento de hisotory y los datos de validacion podemos ver la exactitu del modelo

            # now vamos a guardar los modelos. este modelo es super rapido y potente.  para guardar store date

            ## lets ask if the user wants yo store the model
            r=input('savel model?(Y-N)')
            if r=='Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost')) # esto es para definir donde va la informacion para todos loslugares  toma la carpta donde esta
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.json') # es el nombre del modelo se pondra en otro parametro 
                print(f'Checkpoint path: {checkpoint_file}')
                model.save_model(checkpoint_file)
                print('Model Saved!')

            elif r == 'N':
                print('Model NOT Saved!')

            else:
                print('Command not recognized')
                 # solo los el json, tiene la fonciguraion de entradas de los modelos que usamos o info
        else:
            ##we are not training a model here, just using an already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost')) # para saber donde esta el modelo 
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.json') # con esto vamos a cargar el modelo o cual vamos a cargar.
            ##lets load the model
            model.load_model(checkpoint_file) # los valores numericos. 

            # ahora vamos a predecir los valores de  neuvo 
            ##prediction output
            pred_out = model.predict(train_features)# con todos los valores 
            
             ##Let's denormalize the data
            SC = StandardScaler()

            original_labels_norm = SC.fit_transform(original_labels)

            if neurons == 1:
                pred_out = pred_out.reshape(-1,1)

            pred_out_denorm = SC.inverse_transform(pred_out)

            pred_df = pd.DataFrame(pred_out_denorm)

            result_data = pd.concat([original_features,pred_df],axis=1)

            print(f'Dataframe : {result_data}')
            
            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_file_))),'data','results'))

            results_file = os.path.join(results_dir, f'{chk_name}_RESULTS_XGB.xlsx')

            ##Let's store the dataframe as excel file
            result_data.to_excel(results_file, index = False, engine = 'openpyxl')

            
            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(train_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title(f'Prediction Output of model {chk_name}')
            plt.legend()
            plt.show()

            ##lets show the accuracy value for this training batch
            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction accuracy: {accuracy:.2f}%')



    def build_model(self, n_estimators, learning_rate, verbosity):
        model = xg.XGBRegressor(
            objective='reg:squarederror', ##Loss funcion for training determination
            colsample_bytree=0.5, ##Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)(between 0 - 1)
            learning_rate=learning_rate, 
            n_estimators=n_estimators, ##number of ramifications or branches (neurons)
            reg_lambda=2, ##Makes 2 cuts in the information path to force the training of the whole neural network thus, preventing overfitting
            gamma=0, ##reduces random value for reg_lambda cuts (0-1)
            max_depth=self.depth, ##Number of layers
            verbosity=verbosity, ##Shows debug info in terminal (0 None, 1 Shows info)
            subsample=0.8, ##Randomly splits the data for training for each iteration (0,1)(0-100%)
            seed=20, ##Seed for random value, for reproductibility 
            tree_method='hist', ##ramification methos (Hist: reduces significantly the amount of data to be processed)
            updater='grow_quantile_histmaker,prune'
        )

        return model
    
# Cantidad de estimadores es la cantidad de iteraciones  capas por profundidad
        



    
