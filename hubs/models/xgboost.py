##lets import the basic libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##lets import the neural model libraries
import xgboost as xg

##import the metrics libraries
from sklearn.metrics import accuracy_score as acs

##for denormalizing the data
from sklearn.preprocessing import StandardScaler


class xgb:
    def __init__(self, depth):
        self.depth = depth ##depth of decision tree
        #el atributo de una clase se llama self.algo y luego para llamarlo se usa el mismo self para que sepa que es un atributo de ese metodo

    #CONSTRUCCION DEL MODELO
    def run(self, train_features, test_features, train_labels, test_labels, original_features, original_labels, iter, alfa, stop_condition, chk_name, train, neurons):
        ##lets build the model
        #Toma las neuronas totales, en lugar de tomarlas por cada capa
        ##number of inputs  (for example 13 inputs, i have a depth of 10 n_estimators will be (inputs+1)*depth)
        model = self.build_model((train_features.shape[1]+1)*self.depth, alfa, 1)

        ##metodo para saber que partes son entrenamientos y cuales son prueba. diciendo que el set es evaluacion y el fit es prueba
        eval_set = [(train_features, train_labels),(test_features, test_labels)]

        #XGBOOST NO NECESITA FUNCION DE STOP

        if train:
            ##lets train the model
            #eval_metric: es la metrica de evaluacion. Es lo que se va a graficar (ecm)
            model.fit(train_features, train_labels, eval_metric='mae', eval_set=eval_set, verbose=True)

            ##lets plot results
            history = model.evals_result() #imprime el historial de entrenamiento para ver como sale el entrenamiento y poder revisar cuando salga el plot

            ##print(history)
            train_hist = history['validation_0']['mae'] ## se pone mae, porque es el error promedio, ya que el mse no lo reconoce xgboost

            plt.figure()
            plt.plot(train_hist, 'r', label='Training Loss Function')
            plt.xlabel('Epoch') #iteraciones
            plt.ylabel('mae') #el error cuaadratico promedio
            plt.title('Training History')
            plt.legend()
            plt.show()

            #Vamos a guardar el modelo y mostrar la precision del modelo

            ##validation step. 
            pred_out = model.predict(test_features)

            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(test_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title('Validation')
            plt.legend()
            plt.show()

            ##lets show the accuracy value for this training batch
            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Accuracy: {accuracy:.2f}%')

            ##lets ask if the user wants to store the model. Para saber si el modelo esta bien o no y guardarlo
            r = input('Save model? (Y-N)')
            #Cuando se ha entrenado el modelo, brinda la opcion de guardarlo o no
            if r == 'Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.json')
                print(f'Checkpoint path: {checkpoint_file}')
                model.save_model(checkpoint_file)
                print('Model Saved!')

            elif r == 'N':
                print('Model NOT saved')

            else:
                print('Command not recognized')

        else:
            ## Aqui no se esta entrenando un modelo solo se esta probando uno que el usuario a entregado

            ##we are not training a model here, just using an already existing model
            #os permite ubicar la ruta absoluta de la carpeta de checkpoints, para que se pueda usar desde cualquier carpeta
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost')) 
            #los checkpoints se crean cada vez que se hace un nuevo entrenamiento
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.json')
            ##lets load the model
            model.load_model(checkpoint_file) #carga el archivo jason con el valor numeroco de los pesos y lo ubica en el modelo ya existente 

            ##prediction output
            pred_out = model.predict(train_features) 

            ##lets denormalize the data
            SC = StandardScaler() #objeto para usar la libreria que ayuda a desnormalizar 

            original_labels_norm = SC.fit_transform(original_labels)
            
            if neurons == 1:
                pred_out = pred_out.reshape(-1,1) #convierte el vector a dos dimensiones para usar el fit transform
            
            #necesitamos desnomrmalizar los datos para poder devolver los datos al usuario
            pred_out_denorm = SC.inverse_transform(pred_out)
            # vamos a concatenar (unir) la informacion (las columnas)
            pred_df = pd.DataFrame(pred_out_denorm) #prediccion del dataframe

            result_data = pd.concat([original_features, pred_df], axis=1) #los datos finalles que van a enviarse al usuario

            print(f'Dataframe: {result_data}')


            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'results')) #usar el path absoluto, nos paramos en el directorio raiz

            #print(results_dir)
            results_file = os.path.join(results_dir, f'{chk_name}_RESULTS_XGB.xlsx')

            ##original_features.to_excel('output.xlsx', index=False, engine='openpyxl')

            ##lets store the dataframe as excel file
            #guardar los resultados en excel
            result_data.to_excel(results_file, index=False, engine='openpyxl') #esto es para sacar los datos en un excel


            plt.figure()
            plt.plot(pred_out, 'r', label='Model output')
            plt.plot(train_labels, 'b', label='Real output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title(f'Prediction Output of model {chk_name}')
            plt.legend()
            plt.show()

            
            ##lets show the accuracy value for this training batch
            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100 #se pone train features para entrenar con la mayor parte de la data
            print(f'Prediction accuracy: {accuracy:.2f}%')

    ##metodo build model: será un metodo interno de la clase que no se llamará desde afuera
    def build_model(self, n_estimators, learning_rate, verbosity):
        ## Regresor permite optimizar la prediccion de elementos de series temporables
        model  = xg.XGBRegressor( 
            objective='reg:squarederror', ##Loss funcion for training determination, es la FUNCION DE DESEMPEÑO (error cuadratico medio ecm)
            colsample_bytree=0.5, ##Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)(between 0 - 1)
            ## es decir que de los primeros pesos aleatorios, un porcentaje los vuelve a hacer aleatorios en la siguiente iteracion
            learning_rate=learning_rate, 
            n_estimators=n_estimators, ##number of ramifications or branches (neurons)
            reg_lambda=2, ##Makes 2 cuts in the information path to force the training of the whole neural network thus, preventing overfitting
            gamma=0, ##reduces random value for reg_lambda cuts (0-1)
            max_depth=self.depth, ##Number of layers
            verbosity=verbosity, ##Shows debug info in terminal (0 None, 1 Shows info)
            subsample=0.8, ##Randomly splits the data for training for each iteration (0,1)(0-100%)
            seed=20, ##Seed for random value, for reproductibility 
            tree_method='hist', ##ramification methos (Hist: reduces significantly the amount of data to be processed). Comprime la información de los datos y reduce la canctidad de calculos que se procesan (histogramas)
            updater='grow_quantile_histmaker,prune' #optimizador: permite seleccionar como se va a optimizar 
            ## Entrena de atras hacia adelante: primero entrena las ramificaciones y de ultimo las hojas
            #Toma el 80% de los datos y el 20% lo toma como validacion. Las salidas se actualizan se verifican con la validacion y vuelve a iterar
            #previene overfitting
        )

        return model


