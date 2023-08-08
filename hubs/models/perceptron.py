import numpy as np

class Perceptron:
    def __init__(self):
        pass

    def run (self, train_features,test_features,train_labels,tests_labels,iter):
        print('Training perceptron network....')
        ##here is where all the natural network code is gonna be

        ##training and testing is done here.
        #primero vamos a imprimir las entradas para ver como estan saliendo
#        print(f'TRAIN FEATURES: {train_features}')

        #vamos a  organizar la data
        print(f'features shape: {train_features.shape[1]}')
        Xi= np.zeros((train_features.shape[1]+1,1)) #Input vector, recordar que features=in

        Wij=np.zeros((train_labels.shape[1],train_features.shape[1] + 1 ))#weight matriz

        Aj= np.zeros((train_labels.shape[1],1))#vector de agregacion

        Yk= np.zeros((train_labels.shape[1],1))# vector de salida neuronal

        Yd= np.zeros((train_labels.shape[1],1))#label vector

        Ek= np.zeros((train_labels.shape[1], 1)) # Error vector

        ecm= np.zeros((train_labels.shape[1],1))#ECM vector for each iteration

        ecmT= np.zeros((train_labels.shape[1],iter))#ECM results for rvery iteration= suma de ecm
        #se pone doble parentesis para que el primer parametro sea todo lo que se encuentra entre parentesis
        #ej: train_labels.shape[1],1

        #Fill the wight matrix before training
        for i in range(0, Wij.shape[0]):
            for j in range(0,Wij.shape[1]):
                Wij[i][j]= np.random(-1,1)


        print(train_features[0].transpose())


        for it in range (0,iter):#la cantidad de veces que recorre a train_features
            for d in range(0,train_features.shape[0]):#recorre las filas de train_features
                #pass the data inputs to the input vector Xi
                Xi[0][0]= 1 #Bias
                for i in range(0,train_features.shape[1]):
                    Xi[i+1][1]=train_features[d][i]

                #Lets calculate the agregation for each neuron
                for n in range(0,Aj.shape[0]):
                    for n_input in range(0,Xi.shape[0]):#aqui hacemos el producto punto pa sacar las agregaciones
                        Aj[n][0] = Aj[n][0] + Xi[n_input]*Wij[n][n_input]

                #lets calculate the output for each neuron
                for n in range(0, Yk.shape[0]):
                    if Aj[n][0]< 0: #funcion de activacion
                        Yk[n][0]=0
                    else:
                        Yk[n][0]=1

                #lets calculate the error for each neuron

                #pas train_labels to Yd vector
                for i in range(0, train_labels.shape[1]):
                    Yd[i][0]= train_labels[d][i]

                for n in range(0, Ek.shape[0]):
                    Ek[n][0] = Yd[n][0]-Yk[n][0]
                #lets add the ECM for this data point
                    ecm[n][0]=ecm[n][0] + ((Ek[n][0]^2)/2)

        





