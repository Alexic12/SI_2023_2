import numpy as np
import matplotlib.pyplot as plt


class Perceptron_Multi:
    def __init__(self):
        pass

    def run(
        self,
        train_features,
        test_features,
        train_labels,
        test_labels,
        iter,
        alpha,
        stop_condition,
    ):
        print("Entrenando el modelo perceptron...")
        # Código para el modelo perceptron

        # Organizar los datos
        hidden_neurons = (
            train_features.shape[1] + 1
        )  # Número de neuronas en la capa oculta

        Xi = np.zeros((train_features.shape[1], 1))  # Vector de entrada

        Wij = np.zeros((hidden_neurons, train_features.shape[1]))  # Matriz de pesos

        Aj = np.zeros((hidden_neurons, 1))  # Vector de agregación de la capa oculta

        Hj = np.zeros((hidden_neurons + 1, 1))  # Vector de salida de la capa oculta

        Cjk = np.zeros(
            (train_labels.shape[1], hidden_neurons + 1)
        )  # Matriz de pesos de la capa de salida

        Ak = np.zeros(
            (train_labels.shape[1], 1)
        )  # Vector de agregación de la capa de salida

        Yk = np.zeros((train_labels.shape[1], 1))  # Vector de salida

        Yd = np.zeros((train_labels.shape[1], 1))  # Vector de salida deseada

        Ek = np.zeros((train_labels.shape[1], 1))  # Vector de error

        ecm = np.zeros(
            (train_labels.shape[1], 1)
        )  # Error cuadrático medio por iteración

        ecmT = np.zeros((train_labels.shape[1], iter))  # Error cuadrático medio total

        # Inicializar los pesos de la matriz Wij

        for i in range(0, Wij.shape[0]):
            for j in range(0, Wij.shape[1]):
                Wij[i, j] = np.random.uniform(-1, 1)

        print(f"Pesos iniciales de W: {Wij}")

        # Inicializar los pesos de la matriz Wij

        for i in range(0, Cjk.shape[0]):
            for j in range(0, Cjk.shape[1]):
                Cjk[i, j] = np.random.uniform(-1, 1)

        print(f"Pesos iniciales de C: {Wij}")

        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                # Pasar las entradas al vector de entrada Xi
                for i in range(0, train_features.shape[1]):
                    Xi[i] = train_features[d, i]

                # Calcular la agregación de cada neurona Aj
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0, Xi.shape[0]):
                        Aj[n] = Aj[n] + Wij[n, n_input] * Xi[n_input]

                Hj[0] = 1  # Bias

                # Calcular la función de activación de cada neurona Hj
                for n in range(0, hidden_neurons):
                    Hj[n + 1] = 1 / (1 + np.exp(-Aj[n]))

                # Calcular la fuunción de agregación de la capa de salida
                for n in range(0, train_labels.shape[1]):
                    for n_input in range(0, Hj.shape[0]):
                        Ak[n] = Ak[n] + Cjk[n, n_input] * Hj[n_input]

                # Calcular la salida de cada neurona Yk
                for n in range(0, train_labels.shape[1]):
                    Yk[n] = 1 / (1 + np.exp(-Ak[n]))

                # Calcula la salida deseada Yd
                for i in range(0, train_labels.shape[1]):
                    Yd[i] = train_labels[d, i]

                # Calcular el error de cada neurona Ek
                for n in range(0, Yk.shape[0]):
                    Ek[n] = Yd[n] - Yk[n]
                    ecm[n] = ecm[n] + ((Ek[n] ** 2) / 2)

                # Entrenar los pesos

                # Entrenar los pesos de la matriz Cjk
                for n in range(0, Cjk.shape[0]):
                    for h in range(0, Cjk.shape[1]):
                        Cjk[n, h] = (
                            Cjk[n, h] + alpha * Ek[n] * Yk[n] * (1 - Yk[n]) * Hj[h]
                        )

                # Entrenar los pesos de la matriz Wij
                for h in range(0, Hj.shape[0] - 1):
                    for i in range(0, Wij.shape[1]):
                        for o in range(0, Cjk.shape[0]):
                            Wij[h, i] = (
                                Wij[h, i]
                                + alpha
                                * Ek[o]
                                * Yk[o]
                                * (1 - Yk[o])
                                * Cjk[o, h + 1]
                                * Hj[h]
                                * (1 - Hj[h])
                                * Xi[i]
                            )
