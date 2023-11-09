# reconocer imagenes y editenficar 
##lets import libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten

#from keras.layers.core import Dense

##common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for data visualization
import matplotlib.pyplot as plt

## import scikit-learn to the metricks
from sklearn.metrics import accuracy_score as acs

# todo el comportamiento estara basado en tf 

class conv_tf:
    def __init__(self):
        pass
    def run(self,train_images,test_images,train_labels,test_labels,iter):

        #Respache images to specify that its a singles channel

        train_images =train_images.reshape(train_images.shape[0],28,28,1) # los canales son los colores, solo queremos ver en escala de grices, 28 es el tamano de la imagen
        test_images =test_images.reshape(test_images.shape[0],28,28,1) # valores 255 es blanco dividir la imagen 255

        ##lets normalize the image  dividir por el numero mayor 

        train_images = train_images/255.0 # resepresntada entre 0 y 1
        test_images = test_images/255.0

        ##lets graph some imagen as example 

        #plt.figure(figsize=(10,2))
        #for i in range(5):
            #plt.subplot(1,5,i+1)
           # plt.imshow(train_images[i].reshape(28,28),cmap=plt.cm.binary)
           # plt.xlabel(train_labels[i])
        #plt.show()

        ##lets build the model 

        model = self.build_model()

        ##lets train the model
        history=model.fit(train_images,train_labels,epochs=iter)

        ##lets show the training history 

        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'],'r',label='mse')
        plt.show()

        ##lets show the acurrancy of the model

        test_loss,test_acc = model.evaluate(test_images,test_labels)
        print(f'Model accuracy ={test_acc}')

        def build_model(self):
            model = keras.Sequential()
            ############################convolutional layes (Imagen filtering and feature extraction)
            ##lets add 32 convolutional filter with 3x3  
            #en la primera capa we use the  nearts multiple of the image size from (2, 4, 5,26, 32)
            model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1))) # filtros convolucionales troncos de la imagen y volver uno solo 
            # siempre se pone una segunda capa de filtros siempre el doble de la primera.
            #lets add 64 convolutional
            model.add(Conv2D(64,kernel_size=(3,3),activation='relu')) # codificar toda la informacion de la imagen.

            # ahora estraer las caracteristicas. 
            #for feature extraction we use what is called pooling extraer caracteristicas 

            ##lets extract the features via pooling (pool size determines a matriz size for feature extractopn ) 2*2=4 caracteristicas
            # Mejor empezar con el mas pequeno. o crecer el pool es suficente de 2*2 es 4 

            model.add(MaxPooling2D(pool_size=(2,2)))

            #lets randomly turn on and off neurons to improve generalozation(0-1)(0-100%)

            model.add(Dropout(0.25)) # todos los datos estan procesados , escalizados hasta aqui 

            ##lets flatten the information so we can feed it to a normal neuronal network

            model.add(Flatten()) #ya esta en un vector coloumna 
            ############################convolutional layes (Imagen filtering and feature extraction)
            model.add(Dense(128,activation='relu'))

            ##lets add another layer 
            model.add(Dense(10,activation='softmax'))

            ## ahora toca copilar el modelo 
            model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=['mse'])

            return model
        

        print(train_images)
        #print(train_images.shape[0])
        #print(train_images.shape[1])