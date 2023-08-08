from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron 


class Neural:
    def _int_(self):
        pass

    def run_model(self,model,file_name,iter):
        data = Data() # crear un objeto pra los atributos  tenemos datos numericos, tambien puede ser string entonces toca transformar a datos , toca codificar los string en datos 
        Train_features, test_features, train_label, test_labels = data.data_process(file_name)
        if model == 'perceptron':
            print('Running Perceptron Model')
            #Code for the perceptron model no vamos a usar libreriar.
            P = Perceptron()
            P.run(Train_features, test_features, train_label, test_labels,iter)
        elif model == 'ffm':
            print('Running FFM Model')
            #Code for the ffm model 