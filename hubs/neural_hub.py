
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import PerceptronMulti
from hubs.models.ffm_tf import ffm_tf
from hubs.models.xgboost import xgb
from hubs.models.conv_tf import conv_tf

class Neural:# aqui Creo la clase donde tendre mis diferentes modelos y los parametros que le estra introduciendo. 
    def __init__(self):
        pass
    def run_model(self, model, file_name, iter, alfa, test_split, norm, stop_condition, neurons, avoid_cols,chk_name,train): # Aqui defini cual modelo correr
        data = Data() #Aqui traigo la informacion de los datos ya listo para trabajar 
        if model == 'conv_tf':
            train_images,test_images,train_labels,test_labels = data.download_database('MNIST')
        else:
            train_features, test_features, train_labels, test_labels = data.data_process (file_name, test_split, norm, neurons, avoid_cols)
        if model == 'perceptron': # aqui llamo al modelo perceptron 
            print('Running perceptron model')
            ##Code for the perceptron model
            P = Perceptron()
            P.run (train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)
            
        elif model == 'ffm_tf':
            print('Running ffm_tf')
            ##Code for the perceptron model
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition,chk_name,train)

        elif model == 'perceptron_multi': # aqui llamo al modelo perceptron
            print('Running perceptron Multi Model')
            P = PerceptronMulti()
            P.run (train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition)

        elif model == 'xgb':
            print('Running XGBoost model')
            P = xgb(depth = 10)
            P.run(train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition,chk_name,train)

        elif model == 'conv_tf':
            print('Running conv_tf')
            P = conv_tf()
            P.run(train_images,test_images,train_labels,test_labels,iter)