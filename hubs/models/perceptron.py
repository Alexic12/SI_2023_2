import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        pass

    def run(
        self, train_features, test_features, train_labels, test_labels, iter, alpha
    ):
        print("Entrenando el modelo perceptron...")
        # Código para el modelo perceptron

        # Organizar los datos

        Xi = np.zeros((train_features.shape[1] + 1, 1))  # Vector de entrada

        Wij = np.zeros(
            (train_labels.shape[1], train_features.shape[1] + 1)
        )  # Matriz de pesos

        Aj = np.zeros((train_labels.shape[1], 1))  # Vector de agregación

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

        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                # Pasar las entradas al vector de entrada Xi
                Xi[0] = 1
                for i in range(0, train_features.shape[1]):
                    Xi[i + 1] = train_features[d, i]

                # Calcular la agregación de cada neurona Aj
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0, Xi.shape[0]):
                        Aj[n] = Aj[n] + Wij[n, n_input] * Xi[n_input]

                # Calcular la salida de cada neurona Yk
                for n in range(0, Yk.shape[0]):
                    if Aj[n] >= 0:
                        Yk[n] = 1
                    else:
                        Yk[n] = 0

                # Calcula la salida deseada Yd
                for i in range(0, train_labels.shape[1]):
                    Yd[i] = train_labels[d, i]

                # Calcular el error de cada neurona Ek
                for n in range(0, Ek.shape[0]):
                    Ek[n] = Yd[n] - Yk[n]
                    # ECM para este dato
                    ecm[n] = ecm[n] + ((Ek[n] ** 2) / 2)

                # Entrenar los pesos de la matriz Wij
                for n in range(0, Yk.shape[0]):
                    for w in range(0, Wij.shape[1]):
                        Wij[n, w] = Wij[n, w] + alpha * Ek[n] * Xi[w]

            print(f'Iter: {it}')
            for i in range(0, Yk.shape[0]):
                print(f'ECM {i}: {ecm[i]}')

            # Calcular el error cuadrático medio total
            for n in range(0, Yk.shape[0]):
                ecmT[n, it] = ecm[n]
                ecm[n] = 0

        # Graficar el error cuadrático medio total
        for n in range(0, Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n, :], "r", label="ECM Neurona " + str(n))
            plt.show()
