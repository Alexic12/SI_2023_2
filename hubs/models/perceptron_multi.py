import numpy as np
import matplotlib.pyplot as plt#este es pa graficar.

class Perceptron:
    def __init__(self):
        pass

    def run (self, train_features,test_features,train_labels,test_labels,iter,alfa, stop_condition):
        print('Training perceptron network....')

        #un valor arbitrario, no es absoluto
        hidden_neurons= train_features.shape[1] + 1
        
        Xi= np.zeros((train_features.shape[1],1)) #Input vector, recordar que features=in

        Wij=np.zeros((hidden_neurons,train_features.shape[1]))#weight hidden matriz

        Aj= np.zeros((hidden_neurons,1))# agregation vector for the hidden layer

        Hj=np.zeros((hidden_neurons + 1, 1))# se agrega el bias para las entradas de las nuevas neuronas

        Cjk=np.zeros((train_labels.shape[1], Hj.shape[0]))#weights for output layer
        #Hj es vertical ent tomamos las filas  que tiene ese vector Hj y quedan como las columnas de Cjk
        # lo mismo para train labels como esta horizontal y son las ultimas 2 columnas, ellas quedan
        #como las filas de la matriz Cjk
        Ak=np.zeros((train_labels.shape[1],1))#agrgacion para output layer

        Yk= np.zeros((train_labels.shape[1],1))#activation funticon for output layer

        Yd= np.zeros((train_labels.shape[1],1))#label vector

        Ek= np.zeros((train_labels.shape[1], 1)) # Error vector for output layer

        ecm= np.zeros((train_labels.shape[1],1))#ECM vector for each iteration

        ecmT= np.zeros((train_labels.shape[1],iter))#ECM results for rvery iteration= suma de ecm

        #Fill the Wij wight matrix before training
        for i in range(0, Wij.shape[0]):
            for j in range(0,Wij.shape[1]):
                Wij[i][j]= np.random.uniform(-1,1)#pesos aleatorios de la primera fila y ya


        print(f'W: {Wij}')

        #Fill the Cjk wieght matrix before training
        for i in range(0, Cjk.shape[0]):
            for j in range(0,Cjk.shape[1]):
                Cjk[i][j]= np.random.uniform(-1,1)#pesos aleatorios de la primera fila y ya


        print(f'W: {Cjk}')

        for it in range (0,iter):#la cantidad de veces que recorre a train_features
            for d in range(0,train_features.shape[0]):#recorre las filas de train_features
                #pass the data inputs to the input vector Xi
                for i in range(0,train_features.shape[1]):
                    Xi[i][0]=train_features[d][i]

        #Lets calculate the agregation for each neuron
                for n in range(0,Aj.shape[0]):#itera cada neurona
                    for n_input in range(0,Xi.shape[0]):#aqui hacemos el producto punto pa sacar las agregaciones
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]
                        #peso por entrada da una agregacion de la primera neurona y pasa a la segunda neurona

                #lets calculate the hidden activation funtion for each neuron
                Hj[0][0] = 1 #agregamos bias
                for n in range(0, hidden_neurons):
                    Hj[n+1][0]=1/(1+np.exp(-Aj[n][0]))

                #Lets calculate the agregation funtion for each output neuron
                for n in range (0, train_labels.shape[1]):
                    for n_input in range(0,Hj.shape[0]):
                        Ak[n][0]= Ak[n][0]+ Hj[n_input][0]*Cjk[n][n_input]

                #Lets calculate the activation funtion of the output layer
                for n in range(0,train_labels.shape[1]):
                    Yk[n][0] = 1/(1+np.exp(-Ak[n][0]))

                #Lets fill the Yd array with the labels for this data point
                for i in range(0,train_labels.shape[1]):
                    Yd[i][0]= train_labels[d][i]

                #Lets calsculater the error for this data point
                for n in range(0, Yk.shape[0]):
                    Ek[n][0]= Yd[n][0] - Yk[n][0]
                    #lets add the ECM for this data point
                    ecm[n][0]=ecm[n][0] + (((Ek[n][0])**2)/2)

                    #Lets train the weights

                    #Because of back Propagation we first train  Cjk

                    for n in range(0, Yk.shape[0]): #Cjk.shape[0]
                        for h in range(0,Hj.shape[0]):#Cjk.shape[1]
                            Cjk[n][h] = Cjk[n][h] + alfa*Ek[n][0]*Yk[n][0]*(1-Yk[n][0])*Hj[h][0]

                    #Lets train the Wij Weights
                    #right side rows, left side colums
                    for h in range(0, Hj.shape[0] - 1):# Also it can be hidden_neurons
                        for i in range(0, Xi.shape[0]):
                            for o in range(0, Yk.shape[0]):
                                Wij[h][i]= Wij[h][i] + alfa*Ek[o][0]*Yk[o][0]*(1-Yk[o][0])*Cjk[o][h+1]*Hj[h+1]*(1-Hj[h+1])*Xi[i][0]

                    
                    ##Lets reset the Agregation for each neuron
                    Aj[:][:]= 0
                    Ak[:][:]= 0

            ## lets show the iteration we're in and print the ecm for that iteration
            print(f'Iter: {it}')
            for n in range(0,Yk.shape[0]):
                print(f'ECM{n}: {ecm[n][0]}')

            ## Lets store the total ecm for that specific output for this iteration
            for n in range(0, Yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0  #


                    # ##lets check the stop_condition
                    # flag_training = False
                    # for n in range(0, Yk.shape[0]):
                    #     if ecmT[n][it] <  stop_condition:
                    #         flag_training = True

                    # if flag_training == False :
                    #     it = iter -1 
                    #     break



        for n in range(0, Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:],'r',label = f'ECM Neurona{n}')
            plt.show() 


