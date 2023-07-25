from hubs.data_hub import Data
from models.perceptron import Perceptron


class Neural: 
    def __init__(self) -> None:
        pass

    def run_model (self,model, file_name):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(file_name)
        if model =='perceptron':
            print('Running Perceptron Model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels)

        elif model == 'ffm':
            print('Running FFM Model')
            ##Code for thr perceptron model