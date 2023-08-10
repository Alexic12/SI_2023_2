import numpy as np

class Perceptron:
    def __init__(self):
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter):
        print('Training perceptron network...')

        ##Neural network code
        Xi = np.zeros((train_features.shape[1]+1, 1)) #Input verctor

        Wij = np.zeros((train_labels.shape[1], train_features.shape[1]+1)) #Weight matrix

        Aj = np.zeros((train_labels.shape[1], 1)) #Agregation vector

        Yk = np.zeros((train_labels.shape[1], 1)) #Neural output vector

        Yd = np.zeros((train_labels.shape[1], 1)) #Label vector

        Ek = np.zeros((train_labels.shape[1], 1)) #Error vector

        ecm = np.zeros((train_labels.shape[1], 1)) #ECM vector for each iteration

        ecmT = np.zeros((train_labels.shape[1], iter)) #ECM total for all the iterations

        ##Fill the wight Matrix before training
        for i in range(0, Wij.shape[0]):
            for j in range(0, Wij.shape[1]):
                Wij[i][j] = np.random(-1, 1)

        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                #pass the data inputs to the input vector Xi
                Xi[0][0] = 1 ##Bias - Valor semilla que siempre es 1
                
                for i in range(0 ,train_features.shape[1]):
                    Xi[i+1][0] = train_features[d][i]

                ##Lets calculate the agregation  for each neuron - Es un producto punto
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0, Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Aj[n_input]*Wij[n][n_input]
                
                #Lets calculate the output for each neuron 
                for n in range(0, Yk.sahpe[0]):
                    if Aj[n][0] < 0:
                        Yk[n][0] = 0
                    else:
                        Yk[n][0] = 1
                
                #Lets calculate the error for aech neuron

                #Pass train_labels to Yd vector
                for i in range(0, train_features.shape[1]):
                    Yd[i][0] = train_features[d][i]

                for n in range(0, Ek.shape[0]):
                    Ek[n][0] = Yd[n][0] - Yk[n][0]

                    ##Lets add the ECM for this data point
                    ecm[n][0] = ecm[n][0] + ((Ek[n][0]^2)/2)

        ##Training and testing is done here
        #print(f'Train features :{train_features}')