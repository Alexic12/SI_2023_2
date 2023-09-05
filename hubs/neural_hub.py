
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multicapa import Perceptron_Multicapa




class Neural:
    def __init__(self):
        pass
    
    def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,neurons,avoid_col):
        data = Data()
        train_features,test_features,train_labels,test_labels= data.data_process(file_name,test_split,norm,neurons,avoid_col)
        
        if model == "perceptron":
            print("Running Perceptron Model")
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition)
            
        elif model == "ffm":
            print("Running FFM Model")
            ##Code for FFM model
            
        elif model == "perceptron_multicapa":
            print("Running Perceptron Multicapa Model")
            P = Perceptron_Multicapa()
            P.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition)