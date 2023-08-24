import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulti :
    def __init__(self) :
        pass

    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        print('Training perceptron network....')


        
        ##Here is where all the neural network code is gonna be

        # Lets organize the data 


        hidden_neurons = train_features.shape[1]+1
        ##Training and testing is done here
        print(f'features shape:{train_features.shape[1]}')
        Xi = np.zeros((train_features.shape[1], 1))

        Wij = np.zeros((hidden_neurons, train_features.shape[1])) #matriz de pesos

        
        Aj = np.zeros((hidden_neurons,1)) #agregation vector

        Hj = np.zeros((hidden_neurons+1,1)) ##activation function for hidden layer

        Cjk = np.zeros((train_labels.shape[1], Hj.shape[0]))  ##weights for output layer

        Ak = np.zeros((train_labels.shape[1], 1))   ##agregation function for output layer

        Yk = np.zeros((train_labels.shape[1],1)) ##activation function for output layer 

        Yd = np.zeros((train_labels.shape[1], 1)) ##label vector 

        Ek = np.zeros((train_labels.shape[1],1)) ##error for putput layer

        ecm = np.zeros ((train_labels.shape[1],1)) #ECM vector for each iteration

        ecmT = np.zeros((train_labels.shape[1], iter)) # ECM results for every iteration

        #fill the Wij weight matrix before training
        for i in range (0, Wij.shape [0]):
            for j in range (0, Wij.shape[1]):
                Wij[i][j] = np.random.uniform(-1,1)

        print(f'W: {Wij}')

        #fill the Cij weight matrix before training
        for i in range (0, Cjk.shape [0]):
            for j in range (0, Cjk.shape[1]):
                Cjk[i][j] = np.random.uniform(-1,1)

        print(f'Cjk: {Cjk}')

        for it in range (0,iter):
            for d in range (0,train_features.shape[0]):
                ##pass the data inputs to the input vectir Xi
                for i in range (0,train_features.shape[1]):
                    Xi[i][0] =train_features[d][i]  


            ##Lets calculate the agregation for each neuron
            for n in range (0, Aj.shape[0]):
                for n_input in range (0, Xi.shape[0]):
                    Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

            Hj[0][0] = 1 ##Bias

            ##Lets calculate the hidden activation function for each neuron 
            for n in range (0, hidden_neurons):
                Hj[n+1][0] = 1/(1*np.exp(-Aj[n][0]))




            ##Lets calculate the agregation function for each output neuron
            for  n in range (0 , train_labels.shape[1]):
                for n_input in range (0, Hj.shape[0]):
                    Ak[n][0] = Ak[n][0]+ Hj[n_input][0]*Cjk[n][n_input]


            ##Lets calculate the activation function of the output layer
            for n in range(0, train_labels.shape[1]):
                Yk[n][0] = 1/(1*np.exp(-Ak[n][0]))

            
            ##Lets fill the Yd array with the labels for this data point


            for i in range (0, train_labels.shape[1]):
                Yd[i][0] = train_labels[d][i]

            ##Lets calcultate the error for this data point
            for n in range ( 0, Yk.shape[0]):
                    Ek[n][0] = Yd[n][0]- Yk[n][0]
                    ##Lets add the ECM for this data point
                    ecm[n][0] = ecm [n][0]+ ((Ek[n][0]**2)/2)

            ##Lets train the weights 


            ## Because of back propragation we first train cjk

            for n in range (0, Yk.shape[0]):##Cjk.shape[0]
                for h in range (0, Hj.shape[0]): ##Cjk.shape[1]
                    Cjk[n][h] = Cjk[n][h]+ alfa*Ek[n][0]* Yk[n][0]* (1-Yk[n][0])*Hj[h][0]

            ##Lets train the Wij weights
            for h in range (0, Hj.shape[0] -1 ):
                for i in range ( 0, Xi.shape[0]):
                    for o in range(0, Yk.shape[0]):
                        Wij[h][i] = Wij[h][i] + alfa*Ek[o][0]*Yk[o][0]*Yk[o][0]*(1-Yk[o][0])*Cjk[o][h+1]*Hj[h+1]*(1-Hj[h+1])*Xi[i][0]

                


            

        

                



                





