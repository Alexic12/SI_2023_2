##lets import libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

#from keras.layers.core import Dense

##common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for data visualization
import matplotlib.pyplot as plt

##import the metrics libraries
from sklearn.metrics import accuracy_score as acs

class conv_tf:
    def __init__(self):
        pass

    def run(self, train_images, test_images, train_labels, test_labels, iter):

        #procesar la data

        ##Respahe images to specify that it's a single channel - Debemos especificar que es solo un canal de imagenes
        # para que solo tiene una matriz de datos, escala de grises
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) # la imagen siempre viene de tamaño 28x28, lo dejamos por defecto de ese tamaño
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

        ##lets normalize the image, se divide entre 255 para que queden representada entre 0 y 1

        #cada pixel tiene un byte de informacion, dando 255 de valores maximos de intensidad de color
        train_images = train_images /255.0
        test_images = test_images /255.0

        ##lets graph some images as example
        plt.figure(figsize=(10,2)) #imprimir una imagen
        
        for i in range(5):
            
            plt.subplot(1,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i].reshape(28,28), cmap=plt.cm.binary) #mostrar el entrenamiento, imprime une imagen binaria
            plt.xlabel(train_labels[i])
        plt.show()

        ##lets build the model
        model = self.build_model()


        ##lets train the model - historial de entreamiento
        history = model.fit(train_images, train_labels, epochs=iter)

        ##lets show the training history
        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()


        ##lets show the acuraccy of the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Model accuracy = {test_acc}')

    
    def build_model(self):

        #PRIMERA CAPA: ARQUITECTURA DE LA RED de una red Convolucional

        #otra forma de crear un modelo con () para que sepa que es secuencial
        model = keras.Sequential()

        #el filtro convolucional va reduciondo el tamaño de las porciones de caracteristicas
        ##########CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)
        ##in the first layer, we use the nearest multiple of the image size from (2, 4, 8, 16, 32, 64, .....)
        
        ##Lets add 32 convolutional filters with 3x3 kernel, because is the closer number to the size 28 that we have
        #KERNEL: se encarga de encontrar patrones de una imagen
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

        ##is normal to add a second layer of convolutional filters the double of the size
        ##Lets add 64 convolutional filters with 3x3 kernel
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

        #POOLING: Es la otra capa de redes convolucionales que permite reunir las caracteristicas 

        ##for feature extraction we use what is called pooling
        ##lets extract the features via pooling (pool size determines a matrix size for feature extraction) 2x2 = 4 features
        #poolsize: numero de caracteristicas que puede extraer 
        model.add(MaxPooling2D(pool_size=(2,2)))

        ##lets randomly turn on and off neurons to improve generalization (0-1) (0-100%)
        model.add(Dropout(0.25))

        
        ##lets flatten the information so we can feed it to a normal neural network
        model.add(Flatten()) #convertimos en un vector columna para ingresarse en la red neuronal normal

        #CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)

        ##its normal to put the number of neurons the double size of the biggest convolutional filter
        model.add(Dense(128, activation='relu')) #relu entrega una informacion continua entre 0 y 1

        ##lets add another layer for the categories we have 10 categories so we add 10 neurons
        model.add(Dense(10, activation='softmax')) #neuronas de salida, n neuronas que son las n caracteristicas que vamos a sacar (en este caso los numeros de 0 a 9)

        #compilar el modelo
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mse'])

        return model
