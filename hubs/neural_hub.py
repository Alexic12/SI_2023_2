from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptronMC import PerceptronMC
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm import ffm_tf

class Neural:
    def __init__(self):
        pass
    
    #def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,nfl,neurons,avoid_col): PARA CONTROLAR NEURONAS OCULTAS
    def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,nfl,neurons,avoid_col):
        data = Data()
        train_features, test_features, train_labels,test_labels = data.data_process(file_name,test_split,norm,neurons,avoid_col)
        if model == 'perceptron':
            print('Running Perceptron Model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels,test_labels,iter,alfa,stop_condition)

        elif model == 'ffm_tf':
            print('Running FFM Model')
            FFM = ffm_tf()
            FFM.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition)
            ##Code for the ffm model

        elif model == 'PMC':
            print('Running Multi Layer Perceptron Model')
            ##Code for the Multi Layer perceptron model
            PMC = PerceptronMulti()
            PMC.run(train_features, test_features, train_labels,test_labels,iter,alfa,stop_condition,nfl)
            


    