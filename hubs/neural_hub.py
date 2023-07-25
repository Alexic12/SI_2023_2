from hubs.data_hub import Data
from models.perceptron import Perceptron


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name):
        data= Data()
        train_features, test_features, train_labels, test_labels = data.data_process('ETHEREUM_PRICE.xlsx')
        if model == 'perceptron':
            print('Running Preceptron Model')

            ##Code for the preceptron model
            P = Perceptron()
            P.run(self, train_features, test_features, train_labels, test_labels)

        elif model == 'ffm':
            print('Running ffm Model')
            
            ##Code for the preceptron model

