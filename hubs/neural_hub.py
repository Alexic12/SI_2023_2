from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron#se hace desde hubs porque hay veces que no guarda el path bien
#Desde neural hub se llama siempre a DATA HUB
from hubs.models.perceptron_multi import Perceptron

class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name,iter,alfa,test_split, norm, stop_condition,neurons,avoid_col):
        data= Data()
        train_features, test_features, train_labels, test_labels =data.data_process (file_name, test_split, norm,neurons,avoid_col)
        if model == 'perceptron':
            print('Running Perceptron Model')
            ##Code for the perceptron model
            P=Perceptron()
            P.run(train_features, test_features, train_labels, test_labels,iter,alfa, stop_condition)

        elif model == 'ffm':
            print('Running FFM model')
            ##code for the perceptron model

        elif model == 'perceptron_multi':
            print('Running Perceptron multi Model')
            
            P=Perceptron()
            P.run(train_features, test_features, train_labels, test_labels,iter,alfa, stop_condition)

