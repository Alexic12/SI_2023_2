import numpy as np
class Perceptron:
    def __init__(self):
        pass
    
    def run(self, train_features, test_features, train_labels, test_labels, iter):
        print("Entrenar Perceptron")
        #Organizar Datos
        Xi = np.zeros((train_features.shape[1]+1, 1)) #Vector Entradas

        Wij = np.zeros((train_labels.shape[1], train_features.shape[1]+1))#Matriz pesos

        Aj = np.zeros((train_labels.shape[1], 1))#Vector agregacion

        Yk = np.zeros((train_labels.shape[1], 1))#Vector salidas

        Yd = np.zeros((train_labels.shape[1], 1))#Vector Labels

        Ek = np.zeros((train_labels.shape[1], 1))#Vector Error

        ecm = np.zeros((train_labels.shape[1], 1))#Error cuadratico medio

        ecmT = np.zeros((train_labels.shape[1], iter))#ECM total por iteracion

        for i in range(Wij.shape[0]):#Llenar con pesos aleatorios entre -1 y 1
            for j in range(Wij.shape[1]):
                Wij[i][j] = np.random(-1, 1)

        for it in range(iter):
            for d in range(train_features.shape[0]):
                #meter datos a Xi
                Xi[0][0] = 1 #Bias
                for i in range(train_features.shape[1]):
                    Xi[i+1][1] = train_features[d][i]
                #Calcular agregacion para cada neurona
                for n in range(Aj.shape[0]):
                    for n_input in range(Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]
                #calcular output por cada neurona
                for n in range(Yk.shape[0]):
                    if Aj[n][0] < 0:
                        Yk[n][0] = 0
                    else:
                        Yk[n][0] = 1
                #Calcular error
                for i in range(train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]

                for n in range(Ek.shape[0]):
                    Ek[n][0] = Yd[n][0] - Yk[n][0]
                    #ECM para este dato
                    ecm[n][0] = ecm[n][0] + (Ek[n][0]**2)/2
                
                
                