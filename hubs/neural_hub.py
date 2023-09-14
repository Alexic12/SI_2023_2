
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm_tf import ffm_tf
from hubs.models.xgboost import xgb


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons, avoid_col):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(file_name, test_split,norm, neurons, avoid_col)
        if model == 'perceptron':
            print('Running Perceptron Model')
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'ffm_tf':
            print('Running FFM Model')
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'perceptron_multi':
            print('Running perceptron Multi Model')
            P = PerceptronMulti()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'xgb':
            print('Running XGBoost model')
            P = xgb(depth = 10)
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)





