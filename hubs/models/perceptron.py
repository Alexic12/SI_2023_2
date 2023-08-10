import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels, test_labels, iter,alpha):
        print('Train perceptron network....')
        ##here is where all neural network code is gonna be 
        
        #Lets organiuze the data 
        xi = np.zeros((train_features.shape[1] + 1,1)) #input vector 
        wij = np.zeros((train_labels.shape[1],train_features.shape[1] + 1)) #weight matrix 
        Aj = np.zeros((train_labels.shape[1],1)) #agregation vector 
        yk = np.zeros((train_labels.shape[1],1)) #neural output vector 
        yd = np.zeros((train_labels.shape[1],1)) #labels vector
        ek = np.zeros((train_labels.shape[1],1)) #error  vector
        ecm = np.zeros((train_labels.shape[1],1)) #ECM  vector for each iteration
        ecmT = np.zeros((train_labels.shape[1],iter)) #ECM results for every iteration 
        
        #print(train_features[0].transpose())
        
        #Fill the weight matrix before training 
        for i in range(0,wij.shape[0]):
            for j in range(0,wij.shape[1]):
                wij[i][j] = np.random.uniform(-1,1)
                
        
        for it in range(0, iter):
            for d in range(0, train_features.shape[0]):
                ##pass the data inputs to the input vector xi
                xi[0][0] = 1 #Bias 
                for i in range(0, train_features.shape[1]):
                    xi[i+1][0] = train_features[d][i]
                
                
                ##lets calculate the agregation for each neuron 
                for n in range(0,Aj.shape[0]):
                    for n_input in range(0,xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + xi[n_input]*wij[n][n_input]       
                
                
                ##lets calculate the output for each neuron 
                for n in range (0,yk.shape[0]):
                    if Aj[n][0] < 0:
                        yk[n][0] = 0
                        
                    else:
                        yk[n][0] = 1
                        
                ##lets calculate the error for each neuron 
                
                ##pass train_labels to yd vector 
                for i in range(0,train_labels.shape[1]):
                    yd[i][0] = train_labels[d][i]
                
                for n in range(0,ek.shape[0]):
                    ek[n][0] = yd[n][0]-yk[n][0]
                    ecm[n][0] = ecm[n][0] + ((ek[n][0]**2)/2)
                    
                ##lets add the ECM for this data point
                
                #weight training 
                
                for n in range(0,yk.shape[0]):
                    for w in range(0,wij.shape[1]):
                        wij[n][w] = wij[n][w] + alpha*ek[n][0]*xi[w][0]
                        
            print(f'Iter: {it}')
            for n in range(0,yk.shape[0]):
                print(f'ECM {n} : {ecm[n][0]}')
                    
                        
            for n in range(0,yk.shape[0]):
                ecmT[n][it] = ecm[n][0]
                ecm[n][0] = 0
                
        for n in range(0,yk.shape[0]): 
            plt.figure()
            plt.plot(ecmT[n][:], 'r', label = f'ECM neuron {n}')
            plt.show()
                        
                
                    
                    
        ##training and testing is done here 
        
        #print(f'TRAIN FEATURES: {train_features}')
        