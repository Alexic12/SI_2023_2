from hubs.data_hub import data
from hubs.models.perceptron  import Perceptron
from hubs.models.perceptronmulti import Perceptronmulti
#from hubs.models.perceptronprofe  import Perceptron
from hubs.models.ffm_tf import ffm_tf
from hubs.models.xgboost import xgb
from hubs.models.conv_tf import conv_tf

class Neural:
    def __init__(self):
        pass
    
    def run_model(self,model, file_name, iter, alpha, test_split, norm, stop_condition, neurons, avoid_col, chk_name, train, data_type, iden, windows_size, horizon_size):
        Data = data()
        
        if model == 'conv_tf':
            train_images, test_images, train_labels, test_labels = Data.download_database('MNIST')
        
        else:
            
            if data_type == 'time_series':
                
                ##windows_size = 3
                
                ##horizon_size = 1
                
                train_features, test_features, train_labels, test_labels, original_feature, original_labels = Data.time_series_process_adaptative(windows_size, horizon_size, file_name, test_split, norm, iden)
                
            elif data_type == 'data':
            
                train_features, test_features, train_labels, test_labels, original_feature, original_labels = Data.data_process(file_name,test_split,norm,neurons,avoid_col)
        
        if model == 'perceptron':
            print('Running perceptron model')
            ##Code for the perceptron model 
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha,stop_condition)
            
        elif model == 'ffm_tf':
            print('Running FFM model')
            ##Code for the perceptron model
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels, original_feature, original_labels, iter, alpha, stop_condition, chk_name, train, neurons)
            
        elif model == 'perceptron_multi':
            print('Running perceptron_multi model')
            P = Perceptronmulti()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha,stop_condition)
            
        elif model == 'xgb':
            print('Running XGBoost model')
            P = xgb(depth = 10)
            P.run(train_features, test_features, train_labels, test_labels, original_feature, original_labels, iter, alpha, stop_condition, chk_name, train, neurons)
            
        elif model == 'conv_tf':
            print('Running conv_tf model')
            P = conv_tf()
            P.run(train_images, test_images, train_labels, test_labels, iter)
        