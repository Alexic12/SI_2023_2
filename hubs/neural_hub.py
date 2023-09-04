from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import Perceptron_multi

class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(file_name, test_split, norm, neurons)

        if model == 'perceptron':
            print('Corriendo modelo Perceptron')
            ## code for the perceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'ffm':
            print('Corriendo modelo FFM')
            ## code for ffm

        elif model == 'perceptron_multi':
            print('Corriendo modelo perceptron multi')
            ## code for ffm

            P = Perceptron_multi()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)
