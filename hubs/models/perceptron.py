import numpy as np


class Perceptron:
    def __init__(self):
        pass
    
    def run(self, train_features,test_features, train_labels, test_labels,iter):
        print("Training perceptron network.......")
        ##here is where all the nerual network code is gonna be
        
        ##training and testing is done here
        print(f"TRAIN FEATURES: {train_features}")
        
        Xi = np.zeros((train_features.shape[1] + 1,1)) #Input vector
        Wij = np.zeros((train_labels.shape[1],train_features.shape[1] + 1)) #Weight Matrix
        Aj = np.zeros((train_labels.shape[1],1))  #Agregation vector
        Yk = np.zeros((train_labels.shape[1],1)) #Neural Output vector
        Yd = np.zeros((train_labels.shape[1],1)) #Labels vector
        Ek = np.zeros((train_labels.shape[1],1)) #Error vector
        ecm = np.zeros((train_labels.shape[1],1)) #ECM vector for each iteration
        ecmT = np.zeros((train_labels.shape[1],iter))
        
        
        for i in range(0,Wij.shape[0]):
            for j in range(0,Wij.shape[1]):
                Wij[i][j] = np. random(-1,1)
                
                
        for it in range(0,iter) :
            for d in range (0,train_features.shape[0]):
                Xi[0][0] = 1 ##Bias
                for i in range(0, train_features.shape[1]):
                    Xi[i+1][0] = train_features[d][i]
                    
                for n in range(0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[n][0] = Aj[n][0] + Xi[n_input] * Wij[n][n_input]
                        
                for n in range(0,Yk.shape[0]):
                    if Aj[n][0] < 0:
                        Yk[n][0] = 0
                    else:
                        Yk[n][0] = 1
                
                for i in range(0,train_labels.shape[1]):
                    Yd[i][0] = train_labels[d][i]
                     
                for n in range(0,Ek.shape[0]):
                    Ek[n][0] = Yd[n][0]-Yk[n][0]
                    ecm[n][0] = ecm[n][0] + ((Ek[n][0]^2)/2)
                    
                    
                    
                
        