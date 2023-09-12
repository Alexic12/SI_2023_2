from hubs.data_hub import data
from hubs.models.perceptron  import Perceptron
from hubs.models.perceptronmulti import Perceptronmulti
#from hubs.models.perceptronprofe  import Perceptron
from hubs.models.ffm_tf import ffm_tf

class Neural:
    def __init__(self):
        pass
    
    def run_model(self,model,file_name, iter, alpha,test_split,norm,stop_condition,neurons,avoid_col):
        Data = data()
        train_features, test_features, train_labels, test_labels = Data.data_process(file_name,test_split,norm,neurons,avoid_col)
        
        if model == 'perceptron':
            print('Running perceptron model')
            ##Code for the perceptron model 
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha,stop_condition)
            
            
        elif model == 'ffm_tf':
            print('Running FFM model')
            ##Code for the perceptron model
            P = ffm_tf()
            P.run(train_features, test_features, train_labels, test_labels,iter, alpha,stop_condition)
            
        elif model == 'perceptron_multi':
            print('Running perceptron_multi model')
            P = Perceptronmulti()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha,stop_condition)
            

        