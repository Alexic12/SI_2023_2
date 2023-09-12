from hubs.data_hub import Data
from hubs.models.perceptron import Perceptron
from hubs.models.perceptron_multi import Perceptron_Multi
from hubs.models.ffm_tf import ffm_tf


class Neural:
    def __init__(self):
        pass

    def run_model(
        self, model, file_name, iter, alpha, test_split, norm, stop_condition, neurons, avoid_column
    ):
        data = Data()
        train_features, test_features, train_labels, test_labels = data.data_process(
            file_name, test_split, norm, neurons, avoid_column
        )
        if model == "perceptron":
            print("Perceptron")
            P = Perceptron()
            P.run(
                train_features,
                test_features,
                train_labels,
                test_labels,
                iter,
                alpha,
                stop_condition,
            )
        elif model == "perceptron_multi":
            print("Perceptron Multi")
            PM = Perceptron_Multi()
            PM.run(
                train_features,
                test_features,
                train_labels,
                test_labels,
                iter,
                alpha,
                stop_condition,
            )

        elif model == "ffm_tf":
            print("Feed Forward Model")
            # Código para el modelo Feed Forward Model
            P = ffm_tf()
            P.run(
                train_features,
                test_features,
                train_labels,
                test_labels,
                iter,
                alpha,
                stop_condition
            )
