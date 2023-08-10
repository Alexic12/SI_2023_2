from hubs.data_hub import data
from hubs.models.perceptron  import Perceptron

class Neural:
    def __init__(self):
        pass
    
    def run_model(self,model,file_name, iter, alpha,split):
        Data = data()
        train_features, test_features, train_labels, test_labels = Data.data_process(file_name,split)
        
        if model == 'perceptron':
            print('Running perceptron model')
            ##Code for the perceptron model 
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha)
            
            
        elif model == 'FFM':
            print('Running FFM model')
            ##Code for the perceptron model
        