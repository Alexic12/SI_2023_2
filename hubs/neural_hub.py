
from hubs.data_hub import Data
from hubs.models.perceptron_multicapa import Perceptron


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, n_ocultas, n_salidas):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process (file_name, test_split, norm, n_ocultas, n_salidas)
        if model == 'perceptron':
            print('Running perceptron model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run (train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)
            

        elif model == 'ffm':
            print('Running FFM model')
            ##Code for the FFM model
