import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def _init_(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels,iter,alfa,stop_condition):
        print('Training perceptron network.....')
        ##here is where all the neural network code is gonna be
        
        ### LEST ORGANIZE THE DATA
        
        Xi =np.zeros((train_features.shape[1]+1,1))# Input vector

        Wij=np.zeros((train_labels.shape[1],train_features.shape[1]+1))#weight Matrix

        Aj=np.zeros((train_labels.shape[1],1)) #Agregation vector

        Hj=np.zeros((train_labels.shape[1],1))# Salida de la neurona oculta

        Yd =np.zeros((train_labels.shape[1],1))#Label VEctor

        Ek=np.zeros((train_labels.shape[1],1)) #Error vector

        ecm= np.zeros((train_labels.shape[1],1)) #ECM vector for each iteration

        ecmT=np.zeros((train_labels.shape[1],iter))#ECm results for every iteration
        
        Yk=np.zeros((train_labels.shape[1],1))# Salida de la neurona oculta
        
        #multicapa

        Cjk=np.zeros((train_labels.shape[1],train_features.shape[1]+1))# matriz de pesos de neurona de salida

        Ak=np.zeros((train_labels.shape[1],1)) #Agregation vector neurona de salida

        
        ##Fill the wight Matrix before training
        for i in range(0,Cjk.shape[0]):
             for j in range(0, Cjk.shape[1]):
                 Cjk[i][j]=np.random.uniform(-1,1)

        ## Bias 
        for it in range(0,iter):
            for d in range(0,Hj.shape[0]):
                ##pass the data inpputs to the input vector Xi
                Hj[0][0]=1 ##Bias

            #Agragacion de la neurona de salida
            for n in range(0,Ak.shape[0]):
                for n_input in range(0,Hj.shape[0]):
                    Ak[n][0] = Ak[n][0]+ Hj[n_input]*Cjk[n][n_input]

            #Funcion de agragacion
            for n in range(0,Yk.shape[0]):
                if Aj[n][0] < 0:
                    Yk[n][0]=0
                else:
                    Yk[n][0]=1


               

       