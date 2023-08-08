import numpy as np 
class Perceptron:
    def _init_(self):
        pass
    def run(self,Train_features, test_features, train_label, test_labels,iter): ## muestra una grafica del entrenamiento.
        print('Training perceptron network')
        ## here is where all the neural network code is gonna be 

        #lets organize the data 
        Xi = np.zeros((Train_features.shape[1]+1,1)) # las cantidad de columnas # Input vector  ademas se le puos el bios

        Wij =np.zeros((train_label.shape[1],Train_features.shape[1]+1))# las cantidad de neuronas que tengo, es decir cantidad de salidas  matriz de pesos 
        
        Aj = np.zeros((train_label.shape[1],1))
        
        Yk= np.zeros((train_label.shape[1],1)) # Neural Output vector 

        Yd =np.zeros((train_label.shape[1],1)) # Label vector 
        
        Ek = np.zeros((train_label.shape[1],1))# Error vector 

        ecm = np.zeros((train_label.shape[1],1)) #ECM vector for each intertion

        ecmT= np.zeros((train_label.shape[1],iter)) ## ECM results for every iteration

        ##Fill the wight Matrix before training
        for i in range(0,Wij,shape[0]):
           for j in range(0,Wij.shape[1]):
              Wij[i][j] = np.random(-1,1)

        for it in range(0,iter): # para para iteracion  es una pasada completa 
            for d in range(0,Train_features.shape[0]): # va a recoger cada fila. 
                #pass
                Xi[0][0] = 1 #Bias\
                for i in range(0,Train_features.shape[1]):
                    Xi[i+1][0]= Train_features[d][i] # se pone +1 por el bias 

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
                    Yd[i][0]=train_label[d][i] # para oganizar las datos 

                for n in range(0,Ek.shape[0]):
                    Ek[n][0]= Yd[n][0]-Yk[n][0]
                ##lets add the ECM for this data point
                    ecm[n][0]= ecm[n][0]+((Ek[n][0]^2)/2)




                
            
