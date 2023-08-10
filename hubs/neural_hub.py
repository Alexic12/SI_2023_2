from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron


class Neural:
    def __init__(self):
        pass

    def run_model(self, model, file_name, iter, alpha, test_split):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(
            file_name, test_split
        )
        if model == "perceptron":
            print("Perceptron")
            P = Perceptron()
            P.run(train_features, test_features, train_labels, test_labels, iter, alpha)
            # Código para el modelo perceptron

        elif model == "ffm":
            print("Feed Forward Model")
            # Código para el modelo Feed Forward Model
