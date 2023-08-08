from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter):
        data= Data()
        train_features, test_features, train_labels, test_labels = data.data_process('ETHEREUM_PRICE.xlsx')
        if model == 'perceptron':
            print('Running Preceptron Model')

            ##Code for the preceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter)

        elif model == 'ffm':
            print('Running ffm Model')
            
            ##Code for the preceptron model

