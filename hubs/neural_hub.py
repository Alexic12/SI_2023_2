
from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multicapa import Perceptron_Multicapa
from hubs.models.ffm_tf import ffm_tf
from hubs.models.xgboost import xgb





class Neural:
    def __init__(self):
        pass
    
    def run_model(self, model, file_name,iter,alfa,test_split,norm,stop_condition,outputs,avoid_col,chk_name,train):
        data = Data()
        train_features,test_features,train_labels,test_labels, original_features,original_labels= data.data_process(file_name,test_split,norm,outputs,avoid_col)
        
        if model == "perceptron":
            print("Running Perceptron Model")
            ##Code for the perceptron model
            P = Perceptron()
            P.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition)
            
        elif model == "ffm_tf":
            print("Running FFM Model")
            P = ffm_tf()
            P.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition,chk_name)

            
        elif model == "perceptron_multicapa":
            print("Running Perceptron Multicapa Model")
            P = Perceptron_Multicapa()
            P.run(train_features,test_features,train_labels,test_labels,iter,alfa,stop_condition)

        elif model == "xgb":
            print("Running XDGBoost Model")
            P = xgb(depth = 10)
            P.run(train_features,test_features,train_labels,test_labels, original_features,original_labels,iter,alfa,stop_condition,chk_name,train,outputs)