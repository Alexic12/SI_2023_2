import numpy as np

class Perceptron:
    #el modelo recibe todos los datos como los de entrenamiento como los de prueba
    def __init__(self):
        pass

    def run(self,train_feature,test_features,train_labels,test_labels,iter):
        print('Training perceptron network..')

        #here is where all the neural network code is gonna be


        print(f'TRAIN FEATURES: {train_feature}')


        ##Organizar los daoros en una matriz
        #filas es la cantidad de neuronas y columnas son los pesos

        Xi = np.zeros((train_feature.shape[1] + 1,1)) #input vector

        Wij = np.zeros((train_labels.shape[1],train_feature[1] + 1)) #weight matrix

        Aj = np.zeros((train_labels.shape[1],1)) #Agregation vector (tiene el tamaño de neurona)

        Yk = np.zeros((train_labels.shape[1],1)) #Neural output vector

        Yd = np.zeros((train_labels.shape[1],1)) #Label Vector

        Ek = np.zeros((train_labels.shape[1],1)) #Error Vector

        ecm = np.zeros((train_labels.shape[1],1)) #ECM vector for each iteration

        ecmT = np.zeros((train_labels.shape[1],iter)) # Ecm results for every iteration

        print(train_feature[0].transpose())

        #Se debe llenar la matriz W con valores aleatorios antes de hacer el entranamiento
        for i in range(0,Wij.shape[0]): #filas
            for j in range(0,Wij.shape[1]): #columnas
                Wij[i][j] =  np.random(-1,1) 

        #para hacer las iteraciones
        for it in range (0, iter):
            for d in range(0,train_feature.shape[0]): #recorrido de todos los datos
                 #pasar las entradas de datos al vector Xi de entrada
                 Xi[0][0] = 1 #Bias
                 for i in range(1,train_feature.shape[1]):
                     Xi[i+1][1] = train_feature[d][i]

        #Matriz de agregacion (sumatoria peso*entrada)
        #Calcular la agragacion para cada neurona 

        for n  in range(0,Aj.shape[0]):
            for n_input in range(0,Xi.shape[0]):
                Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

        # Calcular la salida de cada neurona
        for n in range (0, Yk.shape[0]): 
            if Aj[n][0] < 0: #funcion de acticacion
                Yk[n][0] = 0
            else:
                Yk[n][0] = 1

        # Calcular el error para cada neurona

        #pasar los train_labels a vector Yd (salida deseada)
        for i in range(0,train_labels.shape[1]):
            Yd[i][0] = train_labels[d][i]

        for n in range (0,Yk.shape[0]):
            Ek[n][0] = Yd[n][0]-Yk[n][0]

        #Error cuadratico medio ECM de cada dato
        ecm[n][0] = ecm[n][0] + ((Ek[n][0])^2/2)




