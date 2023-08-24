import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition, capas):
        print('Training perceptron network...')

        hidden_neurons = train_features.shape[1] + 1  # numero recomendado

        xi = np.zeros((train_features.shape[1], 1)) # input vector

        hj = np.zeros((hidden_neurons + 1, 1))

        wij = np.zeros((hidden_neurons, train_features.shape[1])) # matriz de pesos

        cjk = np.zeros((train_labels.shape[1], hidden_neurons + 1)) # matriz de pesos

        aj = np.zeros((hidden_neurons, 1)) # vector de agregacion

        bk = np.zeros((train_labels.shape[1], 1))

        yk = np.zeros((train_labels.shape[1], 1)) # neural output vector

        yd = np.zeros((train_labels.shape[1], 1)) # label vector

        ek = np.zeros((train_labels.shape[1], 1)) # ecm vector for each iteration

        ecm = np.zeros((train_labels.shape[1], 1)) # ecm vector for each iteration

        ecmT = np.zeros((train_labels.shape[1], iter)) # ecm results for every iteration

        # fill the weight matrix before training
        for i in range(0, wij.shape[0]):
            for j in range(0, wij.shape[1]):
                wij[i][j] = np.random.uniform(-1, 1) # incluye decimales, es mejor asi para evitar cambios muy grandes de tipo de dato despues

        for i in range(0, cjk.shape[0]):
            for j in range(0, cjk.shape[1]):
                cjk[i][j] = np.random.uniform(-1, 1) # incluye decimales, es mejor asi para evitar cambios muy grandes de tipo de dato despues

        print(f'Wij: {wij}')
        print(f'cjk: {cjk}')

        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                ## pass the data inputs to the input vector xi
                hj[0][0] = 1 #bias

                for i in range(0, train_features.shape[1]):
                    xi[i][0] = train_features[d][i]

                # lets calculate the agregation for each neuron
                for n in range(0, aj.shape[0]):
                    for input in range(0, xi.shape[0]):
                        aj[n][0] = aj[n][0] + xi[input]*wij[n][input]

                # lets calculate the output for each neuron
                for n in range(0, hidden_neurons):
                    hj[n+1][0] = 1/(1 + np.exp(-aj[n][0]))

                for n in range(0, train_labels.shape[1]):
                    for input in range(0, hj.shape[0]):
                        bk[n][0] = bk[n][0] + hj[input]*cjk[n][input]

                # lets calculate the output for each neuron
                for n in range(0, yk.shape[0]):
                    yk[n][0] = 1/(1 + np.exp(-bk[n][0]))

                # lets calculate the error for each neuron

                # pass train_labels to yd vector
                for i in range(0, train_labels.shape[1]):
                    yd[i][0] = train_labels[d][i]

                for n in range(0, ek.shape[0]):
                    ek[n][0] = yd[n][0] - yk[n][0]
                    # lets add the acm for this data point
                    ecm[n][0] = ecm[n][0] + ((ek[n][0]**2)/2)

                # weight training 
                for n in range(0, cjk.shape[0]):
                    for w in range(0, cjk.shape[1]):
                        cjk[n][w] = cjk[n][w] + alfa*ek[n][0]*hj[w][0]*yk[n][0]*(1 - yk[n][0])

                for n in range(0, hj.shape[0] - 1):
                    for w in range(0, xi.shape[0]):
                        for o in range(0, yk.shape[0]):
                            wij[n][w] = wij[n][w] + alfa*ek[o][0]*yk[o][0]*(1 - yk[o][0])*cjk[o][n+1]*hj[n+1][0]*(1 - hj[n+1][0])*xi[w]

            print(f'Iter: {it}')

            for n in range(0, yk.shape[0]):
                print(f'ecm {n}: {ecm[n][0]}')

            for n in range(0, yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0

            ## lets check the stop_condition
            flag_training = False
            for n in range(0, yk.shape[0]):
                if ecmT[n][it] != stop_condition:
                    flag_training = True

            if flag_training == False:
                it = iter - 1
                break

        print(f'Wij: {wij}')
        print(f'cjk: {cjk}')
        for n in range(0, yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ecm neurona {n}')
            plt.show()
