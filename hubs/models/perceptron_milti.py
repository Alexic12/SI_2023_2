import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulti:
    def _init_(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels,iter,alfa):
        print('Training multicapa perceptron network.....')
        

        hidden_neurons = train_features.shape[1]+1 # se recomienda iniciar con numero de entradas mas uno, no es absoluto, luego se puede cambiar si no funciona

        Xi =np.zeros((train_features.shape[1]+1,1))# Input vector

        Wij=np.zeros((hidden_neurons,train_features.shape[1]+1)) #weight Matrix

        Aj=np.zeros((hidden_neurons,1)) #Agregation vector de las ocultas

        Hj=np.zeros((hidden_neurons+1,1)) #Las salidas de las ocultas, seran las entradas de la neurona de salida + bias

        Cjk=np.zeros(train_labels.shape[1],Hj.shape[0])

        Ak=np.zeros(train_labels.shape[1],1)

        Yk=np.zeros((train_labels.shape[1],1))

        Yd=np.zeros(train_labels.shape[1],1)

        Ek=np.zeros((train_labels.shape[1]),iter)

        ecm= np.zeros((train_labels.shape[1],1)) #ECM vector for each iteration

        ecmT=np.zeros((train_labels.shape[1],iter))#ECm results for every iteration

        #Fill the wight Matrix before training
        for i in range(0,Wij.shape[0]):
            for j in range(0, Wij.shape[1]):
                Wij[i][j]=np.random.uniform(-1,1)

        print(f'Wij :{Wij}')

        #Fill the wight Matrix before training
        for i in range(0,Cjk.shape[0]):
            for j in range(0, Cjk.shape[1]):
                Wij[i][j]=np.random.uniform(-1,1)

        print(f'Cjk :{Cjk}')

        for it in range(0,iter):
            for d in range(0,train_features.shape[1]):
                #se hace una iteracion a la vez
                for i in range(0,train_labels.shape[1]):
                    Xi[i][0] = train_features[d][1]

                ## Lets calculate the Agregation para las neuronas
                for n in range(0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0] = Aj[n][0]+ Xi[n_input]*Wij[n][n_input]

                # para calcular la funcion de agregacion sigmatica de cada neurona oculta y el Bias
                Hj[0][0]=1
                for n in range(0,hidden_neurons):
                    Hj[n+1][0] = 1/(1+np.exp(-Aj[n][0]))

                #Para calcular la Agragacion de cada neurona de salida
                # este tamaño puede sacarse de Yk, Ak o los train labels porque son del mismo tamaño

                for n in range(0,train_labels.shape[1]):
                    for n_input in range (0,Hj.shape[0]):
                        Ak[n][0] = Ak[n][0] + Hj[n_input[0]*Cjk[n][n_input]]
                        
                #Calcular la funcion de agragacion sigmatica de la neurona de salida
                for i in range(0,train_labels.shape[1]):
                    Yk[n][0] = 1/(1+np.exp(-Ak[n][0]))

                ## BACKPROPAGATION

                for i in range(0,train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]

                #calcular el error en este punto de dato
                for n in range(Yk.shape[0]):
                    Ek[n][0] = Yd[n][0] - Yk[n][0]
                    ecm[n][0]=ecm[n][0]+ (((Ek[n][0])**2)/2)

                #primero se entrena las neuronas de salida con Cjk
                for n in range(0,(Yk.shape[0])): #Cjk.shape[0]
                    for h in range(0,Hj.shape[0]): #Cjk.shape[1]
                        #Esta es la derivada para entrenar los pesos de Cjk 
                        Cjk[n][h] = alfa*Ek[n][0]*Yk[n][0]*[1-Yk[n][0]*Hj[h][0]]

                #Entrenar los pesos Wij en la capa interna 
                for h in range(0,Hj.shape[0]-1): #itera a traves de las neuronas, tambien puede se de 0 a hidden neurons
                    for i in range(0,Xi.shape[0]): #itera a traves de las entradas
                        for o in range(0,Yk.shape[0]): # itera en las salidas
                            Wij[h][i] = Wij[h][i] + alfa*Ek[0][0]*Yk[o][0]*(1-Yk[o][0]*Cjk[o][h]*Cjk[o][h+1]*Hj[h+1]*(1-Hj[h+1])*Xi[i][0])

                


                


        





        
        