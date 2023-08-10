import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    #el modelo recibe todos los datos como los de entrenamiento como los de prueba
    def _init_(self):
        pass

    def run(self,train_feature,test_features,train_labels,test_labels, iter, alfa):
        print('Training perceptron network..')

        #here is where all the neural network code is gonna be


        print(f'TRAIN FEATURES: {train_feature}')


        ##Organizar los daoros en una matriz
        #filas es la cantidad de neuronas y columnas son los pesos

        Xi = np.zeros((train_feature.shape[1]+1 ,1)) #input vector

        Wij=np.zeros((train_labels.shape[1],train_feature.shape[1]+1))#weight matrix

        Aj = np.zeros((train_labels.shape[1],1)) #Agregation vector

        Yk = np.zeros((train_labels.shape[1],1)) #Neural output vector

        Yd = np.zeros((train_labels.shape[1],1)) #Label Vector

        Ek = np.zeros((train_labels.shape[1],1)) #Error Vector

        ecm = np.zeros((train_labels.shape[1],1)) #ECM vector for each iteration

        ecmT = np.zeros((train_labels.shape[1],iter)) # Ecm results for every iteration

        #Llenar los pesos de la matriz antes de entrenar
        for i in range(0,Wij.shape[0]):
            for j in range(0,Wij.shape[1]):
                Wij[i][j] = np.random.uniform(-1,1)

        for it in range(0, iter):
            for d in range (0, train_feature.shape[0]):
                #pasar los datos de entrada al al vetor de entrada Xi
                Xi[0][0] = 1 #Bias
                for i in range(1, train_feature.shape[1]):
                    Xi[i+1][0] = train_feature[d][i]

                #Calcular la agregación de cada neurona
                for n in range (Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0]  = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

                #Calcular la salida para cada neurona
                for n in range(0, Yk.shape[0]):
                    if Aj[n][0]  < 0:
                        Yk[n][0] = 0
                    else:
                        Yk[n][0] = 1

                #Calcular el error para cada neurona

                #Pasar train_label al vector Yd
                for i in range(0, train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]

                for n in range(0, Ek.shape[0]):
                    Ek[n][0] = Yd[n][0]-Yk[n][0]
                    #Calcular error cuadratico medio (ECM)
                    ecm[n][0] = ecm[n][0] + ((Ek[n][0]**2)/2)

                #Pesos de entreno
                for n in range(0, Yk.shape[0]):
                    for w in range(0, Wij.shape[1]):
                        Wij[n][w] = Wij[n][w] + alfa*Ek[n][0]+Xi[w][0]

            print(f"Itereacion{it}")
            for n in range(0, Yk.shape[0]):
                print(f'ECM:{n} : {ecm[n][0]}')
            
            for n in range(0,Yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0

        for n in range(0,Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:],'r',label = f'ECMNurona{n}')
            plt.show()


                    




        