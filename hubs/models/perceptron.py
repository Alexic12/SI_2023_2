import numpy as np
class Perceptron:
    def __init__(self):
        pass
    
    def run(self,train_features, test_features, train_labels, test_labels, it):
        Xi = np.zeros((train_features.shape[1] + 1,1))
        
        Wij = np.zeros((train_labels.shape[1], train_features.shape[1] + 1))
        
        Aj = np.zeros((train_labels.shape[1],1))
        
        Yk = np.zeros((train_labels.shape[1],1))
        
        Yd = np.zeros((train_labels.shape[1],1))
        
        Ek = np.zeros((train_labels.shape[1],1))
        
        ecm = np.zeros((train_labels.shape[1],1))
        
        ecm_total = np.zeros((train_labels.shape[1],it))
        
        for i in range(0,Wij.shape[0]) :
            for j in range(0,Wij.shape[1]):
                Wij[i][j] = np.random(-1,1)
                
        for i in range(0,it):
            for d in range(0,train_features.shape[0]):
                Xi[0][0] = 1
                for j in range(0,train_features.shape[1]):
                    Xi[i+1][0] = train_features[d][i]
                    
                for k in range(0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):
                        Aj[k][0] += Xi[n_input][0]*Wij[k][n_input]
                        
                for k in range(0,Yk.shape[0]):
                    if Aj[k][0]<0:
                        Yk[k][0]= 0
                    else :
                        Yk[k][0]= 1
                
                for k in range(0,train_labels.shape[1]):
                    Yd[k][0] = train_labels[d][k]
                    
                for k in range (0 ,Ek.shape [0]):
                    Ek[k][0] = Yd[k][0] - Yk[k][0]
                    
                    ecm[k][0] += (Ek[k][0]^2)/2