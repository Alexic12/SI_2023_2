
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import PerceptronMulti


class Neural:# aqui Creo la clase donde tendre mis diferentes modelos y los parametros que le estra introduciendo. 
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons, avoid_cols): # Aqui defini cual modelo correr
        data = Data() #Aqui traigo la informacion de los datos ya listo para trabajar 
        train_features, test_features, train_labels, test_labels = data.data_process (file_name, test_split, norm, neurons, avoid_cols)
        if model == 'perceptron': # aqui llamo al modelo perceptron 
            print('Running perceptron model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run (train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)
            
        elif model == 'ffm':
            print('Running FFM model')
            ##Code for the FFM models

        elif model == 'perceptron_multi': # aqui llamo al modelo perceptron
            print('Running perceptron Multi Model')
            P = PerceptronMulti()
            P.run (train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

