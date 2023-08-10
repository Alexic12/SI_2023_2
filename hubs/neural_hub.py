from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron  

class Neural:
    def _init_(self):
        pass

    def run_model(self,model,file_name, iter, alfa):
        data=Data()
        train_features,test_features,train_labels,test_labels=data.data_process(file_name)

        if model=='perceptron':
            print('Runinning Perceptron Model')
            #code for the perceptron model
            P=Perceptron()
            P.run(train_features, test_features,train_labels,test_labels, iter, alfa)

        elif model=='ffm':
            print('Running FFM Model')
            #Code for the perceptron model