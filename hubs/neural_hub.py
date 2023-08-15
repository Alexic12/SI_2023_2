from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron 


class Neural:
    def _int_(self):
        pass

    def run_model(self,model, file_name, iter, alfa, test_split, norm, stop_condition, neurons):
        data = Data() # crear un objeto pra los atributos  tenemos datos numericos, tambien puede ser string entonces toca transformar a datos , toca codificar los string en datos 
        Train_features, test_features, train_label, test_labels = data.data_process(file_name, test_split, norm, neurons)
        if model == 'perceptron':
            print('Running Perceptron Model')
            #Code for perceptron model to don't use premade libraries.
            P = Perceptron()
            P.run(Train_features, test_features, train_label, test_labels,iter,alfa, stop_condition)
        elif model == 'ffm':
            print('Running FFM Model')
            #Code for the ffm model 