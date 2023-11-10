from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptronMC import PerceptronMC
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm import ffm_tf
from hubs.models.xgboost import xgb
from hubs.models.conv_tf import conv_tf


class Neural:
    def __init__(self):
        pass
    
    #def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,nfl,neurons,avoid_col): PARA CONTROLAR NEURONAS OCULTAS
    def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,nfl,neurons,avoid_col,chk_name,train,data_type,identificacion,adapt):
        data = Data()
        if model == 'conv_tf':
            train_images,test_images,train_labels,test_labels = data.download_database('MNIST')
        else:
            if (data_type == 'time_series'):
                window_size = 1
                horizon_size = 1
                if adapt == False:
                    train_features, test_features, train_labels, test_labels, original_features, original_labels = data.timeseries_process(window_size,horizon_size,file_name,test_split,norm,identificacion)  ###Cambiar por adapt
                elif adapt == True:
                    train_features, test_features, train_labels, test_labels, original_features, original_labels = data.timeseries_process_adapt(window_size,horizon_size,file_name,test_split,norm,identificacion)
            elif (data_type == 'data'):
                train_features, test_features, train_labels, test_labels, original_features, original_labels = data.data_process(file_name,test_split,norm,neurons,avoid_col)
        
        if model == 'perceptron':
            print('Running Perceptron Model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features, test_features, train_labels,test_labels,iter,alfa,stop_condition)

        elif model == 'ffm_tf':
            print('Running FFM Model')
            FFM = ffm_tf()
            FFM.run(train_features,test_features,train_labels,test_labels,original_features,original_labels,iter,alfa,stop_condition,chk_name,train,neurons)
            ##Code for the ffm model

        elif model == 'xgb':
            print('Running XGBOOST Model')
            XGB = xgb(depth = 10)
            XGB.run(train_features,test_features,train_labels,test_labels,original_features,original_labels,iter,alfa,stop_condition, chk_name, train,neurons)
            ##Code for the ffm model

        elif model == 'PMC':
            print('Running Multi Layer Perceptron Model')
            ##Code for the Multi Layer perceptron model
            PMC = PerceptronMulti()
            PMC.run(train_features, test_features, train_labels,test_labels,iter,alfa,stop_condition,nfl)

        elif model == 'conv_tf':
            print("Running Convolutional TF Model")
            CTF = conv_tf()
            CTF.run( train_images,test_images,train_labels,test_labels,iter)

            


    