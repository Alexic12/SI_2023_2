import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulti:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        print('Training perceptron network...')
        ##Here is where all the neural network code is gonna be

        hidden_neurons = train_features.shape[1] +1 # Es la formulas que usamos la cual consiste en los input +1 = numero de neuronas ocultas no=ne+1 es el punto de equilibrio
        print('Neuronas ocultas:')
        print(hidden_neurons)
        ##Lets organize the data
        Xi = np.zeros((train_features.shape[1], 1)) #Input vector , entradas de lo datos viene de los train_features
        Wij = np.zeros((hidden_neurons, train_features.shape[1])) #Matriz de pesos first layer las filas se hacen con la cantidad de neuras ocultas y las columnas son los datos de entrada(features)
        Aj = np.zeros((hidden_neurons, 1)) #Agregation vector for hidden
        Hj = np.zeros((hidden_neurons + 1, 1))  #Activation function for hidden layer # aqui le adiconamos el bias Hj=1
        Cjk = np.zeros((train_labels.shape[1], Hj.shape[0])) #Matriz de pesos output layer
        Ak = np.zeros((train_labels.shape[1], 1)) #Agregation function for output layer
        Yk = np.zeros((train_labels.shape[1], 1)) #Activation function output layer #Neural output vector second layer
        Yd = np.zeros((train_labels.shape[1], 1)) #Labels vector
        Ek = np.zeros((train_labels.shape[1], 1)) #Error for output layer
        ecm = np.zeros((train_labels.shape[1], 1)) #Ecm vector for each iteration
        ecmT = np.zeros((train_labels.shape[1], iter)) #Ecm results for every iteration

        #Fill the wight matrix before training
        for i in range(0, Wij.shape [0]):# Con esto recorremos las filas
            for j in range(0, Wij.shape[1]):# Con esto recorremos las columnas 
                Wij[i][j] = np.random.uniform(-1, 1)

        print(f'Wij: {Wij}') # pesos iniciales 

        #Fill de Cjk weight matrix before training
        for i in range(0, Cjk.shape[0]):
            for j in range(0, Cjk.shape[1]):
                Cjk[i][j] = np.random.uniform(-1, 1)# aqui llenamos los pesosaleatorios de cjk
        print(f'Cjk: {Cjk}')

        #Lets calculate the agregation for each neuron
        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):# ojo toca borrar los datos 
                for i in range(0, train_features.shape[1]):
                    Xi[i][0] = train_features[d][i]# aqui llenamos los valores de Xi con la informacion de las entradas features

                #Lets calculate the agregation for each neuron
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0, Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input] # aqui se hace el primer calculo de agregacion de Aj que sale de los las neuronas ocultas

                #Lets calculate the hidden activation function for each neuron
                Hj[0][0] = 1 #Bias
                for n in range(0, hidden_neurons):
                    Hj[n+1][0] = 1/(1+(np.exp(-Aj[n][0])))#aqui se pone la funcion de activacion de las neuronas ocultas. 

                #No descomentar OJO!!
                #for n in range(0, hidden_neurons):
                    #if Aj[n][0] <0:
                        #Hj[n+1][0]=0
                    #else:
                        #Hj[n+1][0]= Hj[n+1][0]
                #No descomentar OJO!!        
                 #No descomentar OJO!!

                #for n in range(0, hidden_neurons):
                   #Hj[n+1][0] = (np.exp(Aj[n][0])-np.exp(-Aj[n][0]))/(np.exp(Aj[n][0])+ np.exp(-Aj[n][0]))
                #No descomentar OJO!! 

                #Lets calculate the agregation function for each output neuron
                for n in range(0, train_labels.shape[1]):
                    for n_input in range(0, Hj.shape[0]):
                        Ak[n][0] = Ak[n][0] + Hj[n_input][0]*Cjk[n][n_input]# aqui se calcula la agregacion de la neurona de salida todo se saca de las parciales.

                #Lets calculate the activation function of the output layer
                for n in range(0, train_labels.shape[1]):
                    Yk[n][0] = 1/(1+(np.exp(-Ak[n][0])))

                #Lets fill the Yd array with the labels for this data point
                for i in range(0, train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]
                    
                ## lets calculate the error for this data point
                print(f'salida {d}')
                for n in range(0, Ek.shape[0]):
                    Ek[n][0] = Yd[n][0]-Yk[n][0]
                    print(f'valor de salida deseadas yd: {Yd[n][0]} valor de salida obtenida Yk{Yk[n][0]}')
                    ##Lets add the ECM for this data point
                    ecm[n][0] = ecm[n][0] + (((Ek[n][0])**2)/2)
                #Lets train the weights

                #Because of back propagation we first train Cjk
                for n in range(0,Yk.shape[0]):
                    for h in range(0, Hj.shape[0]):
                        Cjk[n][h] = Cjk[n][h]+(alfa*Ek[n][0]*Yk[n][0]*(1-Yk[n][0])*Hj[h][0])#recuerda todo lo que esta afectado n mueve la fila en yK 

                #Lets train the Wij weights
                for h in range(0, Hj.shape[0]-1): #Hidden neurons
                    for i in range(0, Xi.shape[0]):
                        for o in range(0, Yk.shape[0]):
                            Wij[h][i] = Wij[h][i]+(alfa*Ek[o][0]*Yk[o][0]*(1-Yk[o][0])*Cjk[o][h+1]*Hj[h+1]*(1-Hj[h+1])*Xi[i][0])

                ##Lets reset the Agregation for each neuron
                Aj[:][:] = 0
                Ak[:][:] = 0

            ##Lets show the iteration we are in and print the ecm for that iteration
            print(f'Iter: {it}')
            for n in range(0, Yk.shape[0]):
                print(f'ECM {n}: {ecm[n][0]}')

            ##Lets store the Total ecm
            for n in range(0, Yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0

            ##Lets check the stop condition
            #flag_training = False
            #for n in range(0, Yk.shape[0]):
                #if ecmT[n][it] != stop_condition:
                    #flag_training = True
                    
            #if flag_training == False:
                #it = iter-1
                #break

        for n in range(0, Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ECM Neurona {n}')
            plt.show()


