import numpy as np 
import matplotlib.pyplot as plt
class Perceptron:
    def _init_(self):
        pass
    def run(self,Train_features, test_features, train_label, test_labels, iter, alfa, stop_condition): ## draw a training graphic.
        print('Training perceptron network')
        ## here is where all the neural network code is gonna be 

        #lets organize the data 
        Xi = np.zeros((Train_features.shape[1]+1,1)) # amount of columns, input vector including bias

        Wij =np.zeros((train_label.shape[1],Train_features.shape[1]+1))# amount of neurons that i have, e.g, amount of weight's matrix outputs 
        
        Aj = np.zeros((train_label.shape[1],1))
        
        Yk= np.zeros((train_label.shape[1],1)) # Neural Output vector 

        Yd =np.zeros((train_label.shape[1],1)) # Label vector 
        
        Ek = np.zeros((train_label.shape[1],1))# Error vector 

        ecm = np.zeros((train_label.shape[1],1)) #ECM vector for each interation

        ecmT= np.zeros((train_label.shape[1],iter)) ## ECM results for every iteration
        flag_Training = False
        ##Fill the wight Matrix before training
        for i in range(0,Wij.shape[0]):
           for j in range(0,Wij.shape[1]):
              Wij[i][j] = np.random.uniform(-1,1)

        for it in range(0,iter): # each iter is one complete cycle
            for d in range(0,Train_features.shape[0]): # take each row. 
                #pass
                Xi[0][0] = 1 #Bias
                for i in range(0,Train_features.shape[1]):
                    Xi[i+1][0]= Train_features[d][i] # +1 beause bias 

                    ## lets calculate the Agregation for each neuron
                for n in range(0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0] = Aj[n][0]+Xi[n_input]*Wij[n][n_input]
                ##lets calculate the output  for each neuron

                for n in range(0,Yk.shape[0]):
                    if Aj[n][0]<0:
                        Yk[n][0]=0
                    else:
                        Yk[n][0]=1
                ##lets calculate the output  for each neuron

                ##Pass train_labels to Yd vector 
                for i in range(0,train_label.shape[1]):
                    Yd[i][0]=train_label[d][i] #to organize data

                for n in range(0,Ek.shape[0]):
                    Ek[n][0]= Yd[n][0]-Yk[n][0]
                ##lets add the ECM for this data point
                    ecm[n][0]= ecm[n][0]+(((Ek[n][0])**2)/2) 

                #Weight Training 
                for n in range(0,Yk.shape[0]):
                    for w in range(0,Wij.shape[1]):
                        Wij[n][w]=Wij[n][w]+ alfa*Ek[n][0]*Xi[w][0] # Training with each W 

                print(f'Iter:{it}')
                for n in range(0,Yk.shape[0]):
                    print(f'ECM{n}:{ecm[n][0]}')



            for n in range(0, Yk.shape[0]):
                ecmT[n][it]=ecm[n][0]           
                ecm[n][0]=0

            ##lets check the stop_condition
            for n in range(0, Yk.shape[0]):
                if ecmT[n][it] != stop_condition:
                    flag_Training = True

            if flag_Training == False: 
                it = iter - 1 
                break

        print(f'W: {Wij}')
        for n in range(0,Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ECM Neurona{n}')
            plt.show()





                
            