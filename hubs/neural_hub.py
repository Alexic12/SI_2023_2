from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import PerceptronMulti


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons,avoid_col):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(file_name, test_split,norm, neurons,avoid_col)
        if model == 'perceptron':
            print('Running Perceptron Model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)


        elif model == 'ffm':
            print('Running FFM Model')
            ##Code for the perceptron model

        elif model == 'perceptron_multi':
            print('Running perceptron Multi Model')

            P = PerceptronMulti()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

