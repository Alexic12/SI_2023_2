import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        print('Training perceptron network...')
        ##Here is where all the neural network code is gonna be

        ##Lets organize the data
        Xi = np.zeros((train_features.shape[1] + 1, 1)) #Input vector
        Wij = np.zeros((train_labels.shape[1], train_features.shape[1]+1)) #Matriz de pesos
        Aj = np.zeros((train_labels.shape[1], 1)) #Agregation vector
        Yk = np.zeros((train_labels.shape[1], 1)) #Neural output vector
        Yd = np.zeros((train_labels.shape[1], 1)) #Labels vector
        Ek = np.zeros((train_labels.shape[1], 1)) #Error vector
        ecm = np.zeros((train_labels.shape[1], 1)) #Ecm vector for each iteration
        ecmT = np.zeros((train_labels.shape[1], iter)) #Ecm results for every iteration

        #Fill the wight matrix before training
        for i in range(0, Wij.shape [0]):
            for j in range(0, Wij.shape[1]):
                Wij[i][j] = np.random.uniform(-1, 1)

        print(f'w: {Wij}')

        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                Xi[0][0] = 1 #Bias
                for i in range(0, train_features.shape[1]):
                    Xi[i+1][0] = train_features[d][i]

                ###Lets calculate the agregation for each neuron
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0, Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

                ##Lets claculate the output for each neuron
                for n in range(0, Yk.shape[0]):
                    if Aj[n][0] < 0:
                        Yk[n][0] = 0
                    else:
                        Yk[n][0] = 1

                ##Lets claculate the error for each neuron

                ##Pass train_labels to Yd vector
                for i in range(0, train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]


                for n in range(0, Ek.shape[0]):
                    Ek[n][0] = Yd[n][0]-Yk[n][0]
                    ##Lets add the ECM for this data point
                    ecm[n][0] = ecm[n][0] + (((Ek[n][0])**2)/2)

                #Weight training
                for n in range(0, Yk.shape[0]):
                    for w in range(0, Wij.shape[1]):
                        Wij[n][w] =  Wij[n][w] + alfa*Ek[n][0]*Xi[w][0]

                print(f'Iter: {it}')
                for n in range(0, Yk.shape[0]):
                    print(f'ECM {n}: {ecm[n][0]}')

            for n in range(0, Yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0

            ##Lets check the stop_condition
            flag_training = False
            for n in range(0, Yk.shape[0]):
                if ecmT[n][it] != stop_condition:
                    flag_training = True
            
            if flag_training == False:
                it = iter-1
                break

        print(f'w: {Wij}')

        for n in range(0, Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ECM Neurona {n}')
            plt.show()





        ##Training and testing is done here