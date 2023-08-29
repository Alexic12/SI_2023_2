import numpy as np
import matplotlib.pyplot as plt

class PerceptronMC():
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels,test_labels,iter,alfa,stop_condition,nfl):
        print('Training Perceptron MultiLayer network......')
        ##Here is where all the neural network code is going to be

        ##Let´s organize the data
        
        Xi = np.zeros((train_features.shape[1],1)) #Input Vector

        Wij = np.zeros((nfl,train_features.shape[1])) #Weight Matrix First Layer

        Aj = np.zeros((nfl,1)) #Agregation Vector

        Hj = np.zeros((nfl + 1,1))

        Cjk = np.zeros((train_labels.shape[1],nfl + 1)) #Weight Matrix Second Layer

        Amc = np.zeros((train_labels.shape[1],1)) #Agregation vector Second Layer

        Yk = np.zeros((train_labels.shape[1],1)) #Neural Output Vector neurons,1

        Yd = np.zeros((train_labels.shape[1],1)) #Labels Vector

        Ek = np.zeros((train_labels.shape[1],1)) #Error Vector

        ecm = np.zeros((train_labels.shape[1],1)) #ECM Vector for each iteration

        ecmT = np.zeros((train_labels.shape[1],iter)) #ECM results for every iteration

        ##Fill the Weight Matrix First Layer before training 
        for  i in range (0,Wij.shape[0]):
            for j in range (0,Wij.shape[1]):
                Wij[i][j] = np.random.uniform(-1,1)
            
        print(f'W: {Wij}')
        
        ##Fill the Weight Matrix Second Layer before training 
        for  i in range (0,Cjk.shape[0]):
            for j in range (0,Cjk.shape[1]):
                Cjk[i][j] = np.random.uniform(-1,1)
            
        print(f'C: {Cjk}')

        for it in range(0,iter):
            for d in range(0,train_features.shape[0]):
                ##pass the data inputs to the vector Xi
                for i in range(0,train_features.shape[1]):
                    Xi[i][0] = train_features[d][i]

                ##Let's calculate the Agregation First Layer for each neuron
                for n in range (0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

                ##Let's calculate the Output for each neuron First Layer
                Hj[0][0] = 1 ##Bias
                for n in range(0,nfl):
                    Hj[n+1][0] = 1/(1 + (np.exp(-Aj[n][0])))

                #Let's calculate the Agregation Second Layer for each neuron
                for nsl in range (0,train_labels.shape[1]):
                    for n_input in range(0,Hj.shape[0]):
                        Amc[nsl][0] = Amc[nsl][0] + Hj[n_input]*Cjk[nsl][n_input]
                    
                ##Let's calculate the Output for each neuron Second Layer
                for nsl in range(0,train_labels.shape[1]):
                    Yk[nsl][0] = 1/(1 + (np.exp(-Amc[nsl][0])))
      
                ##Pass train_labels to Yd vector
                for i in range(0,train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]

                ##Let's calculate the error for each neuron
                for nsl in range (0, Ek.shape[0]):
                    Ek[nsl][0] = Yd[nsl][0] - Yk[nsl][0]

                    ##Let's add the ECM for this data point
                    ecm[nsl][0] = ecm[nsl][0] + ((Ek[nsl][0]**2)/2)
                
                #Weight Training Second Layer
                for nsl in range (0,Yk.shape[0]):
                    for c in range(0,Cjk.shape[1]):
                        Cjk[nsl][c] = Cjk[nsl][c] + alfa*Ek[nsl][0]*Hj[c][0]*Yk[nsl][0]*(1-Yk[nsl][0])  

                #Weight Training First Layer
                for n in range (0,nfl):
                    for w in range(0,Wij.shape[1]):
                        for o in range(0,Yk.shape[0]):
                            Wij[n][w] = Wij[n][w] + alfa*Ek[o][0]*Yk[o][0]*(1-Yk[o][0])*Cjk[o][n+1]*Hj[n+1]*(1-Hj[n+1])*Xi[w][0]

            ##Let's reset the Agregation for each neuron
            Aj[:][0] = 0
            Amc[:][0] = 0
                
            ##Let's show the iteration we're in and print the ecm         
            print(f'Iter: {it}')
            for nsl in range (0,Yk.shape[0]):
                print(f'ECM {nsl}: {ecm[nsl][0]}')
             
            ##Let's store the Total ecm for that specific output for this iterartion
            for nsl in range (0,Yk.shape[0]):
                ecmT[nsl][it] = ecm[nsl][0]
                ecm[nsl][0] = 0  

            ##Let's check the stop_condition
            flag_training = False
            for nsl in range (0,Yk.shape[0]):
                if ecmT[nsl][it] < stop_condition:
                    flag_training = True          

            if flag_training == False:
                it = iter-1
                break 
            
        print(f'W: {Wij}')       
        print(f'C: {Cjk}')     
        for nsl in range (0,Yk.shape[0]):
            plt.figure()
            plt.plot(ecmT[nsl][:],'r', label = f'ECM Neurona {nsl}')
            plt.show()

        ##Training and testing is done here