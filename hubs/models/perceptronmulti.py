import numpy as np
import matplotlib.pyplot as plt

class Perceptronmulti:
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels, test_labels, iter,alpha, stop_condition):
        print('Train perceptron network....')
        
        #hidden_neurons = train_features.shape[1] + 1
        hidden_neurons = train_features.shape[1]
        
        Xi = np.zeros((train_features.shape[1], 1)) #input vector 
        
        Wij = np.zeros((hidden_neurons,train_features.shape[1])) #weight matrix 
        
        Aj = np.zeros((hidden_neurons,1)) #agregation vector 
        
        Hj = np.zeros((hidden_neurons + 1,1)) #neural output vector 
        
        Cjk = np.zeros((train_labels.shape[1],Hj.shape[0]))
        
        Ak = np.zeros((train_labels.shape[1],1))
        
        Yk = np.zeros((train_labels.shape[1],1))
        
        Yd = np.zeros((train_labels.shape[1],1)) #labels vector
        
        Ek = np.zeros((train_labels.shape[1],1))
        
        ecm = np.zeros((train_labels.shape[1],1)) #ECM  vector for each iteration
        
        ecmT = np.zeros((train_labels.shape[1],iter)) #ECM results for every iteration 
        
        for i in range(0,Wij.shape[0]):
            for j in range(0, Wij.shape[1]):
                Wij[i][j] = np.random.uniform(-1,1)
                
        print(f'Wij: {Wij}')
        
        for i in range(0,Cjk.shape[0]):
            for j in range(0, Cjk.shape[1]):
                Cjk[i][j] = np.random.uniform(-1,1)
                
        print(f'Cjk: {Cjk}')
        
        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                ##pass the data inputs to the input vector xi
                for i in range(0, train_features.shape[1]):
                    Xi[i][0] = train_features[d][i]
                    
                ##lets calculate the agregation for each neuron 
                for n in range(0, Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]  
                        
                ##lets calculate the output for each neuron 
                Hj[0][0] = 1 ##bias 
                for n in range(0, hidden_neurons):
                    Hj[n+1][0] = (1/(1+np.exp(-Aj[n][0])))
                    
                
                for n in range(0,train_labels.shape[1]):
                    for n_input in range(0,Hj.shape[0]):
                        Ak[n][0] = Ak[n][0] + Hj[n_input][0]*Cjk[n][n_input]
                        
                for n in range(0, train_labels.shape[1]):
                    Yk[n][0] = (1/(1+np.exp(-Ak[n][0])))
                    
                for i in range(0, train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]
                    
                for n in range(0, Yk.shape[0]):
                    Ek[n][0] = Yd[n][0] - Yk[n][0]
                    ##lets add the ECM for this data point
                    ecm[n][0] = ecm[n][0] + (((Ek[n][0])**2)/2)
                    
                for n in range(0, Yk.shape[0]):
                    for h in range(0, Hj.shape[0]):
                        Cjk[n][h] = Cjk[n][h] + alpha*Ek[n][0]*Yk[n][0]*(1-Yk[n][0])*Hj[h][0]
                        
                for h in range(0, Hj.shape[0]-1):
                    #print(f'Hidden Neuron: {h}')
                    for i in range(0, Xi.shape[0]):
                        for o in range(0, Yk.shape[0]):
                            Wij[h][i] = Wij[h][n] + alpha*Ek[o][0]*Yk[o][0]*(1-Yk[o][0])*Cjk[o][h+1]*Hj[h+1]*(1-Hj[h+1])*Xi[i][0]
                            
                Aj[:][0] = 0
                Ak[:][0] = 0
                               
            print(f'Iter: {it}')
            for n in range(0, Yk.shape[0]):
                print(f'ECM {n}: {ecm[n][0]}')
                
                
            for n in range(0, Yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0
            """   
            ##lets check the stop_condition 
            flag_training = False
            for n in range(0, Yk.shape[0]):
                if ecmT[n][it] == stop_condition:
                    flag_training = True
                    
            if flag_training == False:
                it = iter - 1
                break
            """
            #print(f'W: {Wij}')
        for n in range(0, Yk.shape[0]): 
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ECM neuron {n}')
            plt.show()        
                        
                            
                            
                        
                        
            
                    
                
                    



