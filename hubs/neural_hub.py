from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_mul import Perceptron_mul
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm_tf import ffm_tf

class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, outputs, avoid_col):
        data= Data()
        train_features, test_features, train_labels, test_labels = data.data_process(file_name, test_split, norm, outputs, avoid_col)
        if model == 'perceptron':
            print('Running Preceptron Model')

            ##Code for the preceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        if model == 'perceptron_mul':
            print('Running Preceptron_mul Model')
            Pm = Perceptron_mul()
            Pm.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition, nfl)
        
        if model == 'perceptron_multi':
            print('Running Preceptron_multi Model')
            P_m = PerceptronMulti()
            P_m.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)
             
            ##Code for the preceptron model

        if model == "ffm_tf":
            print("running FFM")
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition) 