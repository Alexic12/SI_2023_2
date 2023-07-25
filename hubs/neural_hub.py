from data_hub import Data
from models.perceptron import Perceptron

class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name):
        data = Data()
        train_features, test_features, train_leables, test_leables = data.data_process(file_name)
        if model == 'perceptron':
            print('Corriendo el modelo Perceptron')
            #Código para correr el modelo Perceptron
            P = Perceptron
            P.run(train_features, test_features, train_leables, test_leables)

        elif model == 'ffm':
            print('Corriendo el modelo FFM')

        
            

