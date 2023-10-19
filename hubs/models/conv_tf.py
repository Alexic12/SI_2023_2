##lets import libraries
## Este programa es el de vision artificial
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

        ##Respahe images to specify that it's a single channel
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) # Agarramos todas las imagenes que
                                                                            # vamos a usar para el entrenamiento
                                                                            # y las volvemos de 28x28 pixeles y
                                                                            # de 1 canal de color, es decir
                                                                            # en escala de grises, 3 seria RGB
                                                                            # train_images.shape[0] es cuantas 
                                                                            # imagenes son
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

        ##lets normalize the image
        train_images = train_images /255.0 # Los canales de color (uno solo aqui) cambian valores entre 0-255, 
                                        # la division anterior convierte estos valores para que den entre 0-1
        test_images = test_images /255.0

        ##lets graph some images as example
        plt.figure(figsize=(10,2)) # Figura de 10 (ancho) por 2 pulgadas
        
        for i in range(5):
            
            plt.subplot(1, 5, i+1) # Vamos a hacer graficos dentro del grafico grande, 1 fila y 5 columnas
                                # en la posicion i+1
            plt.xticks([]) # Quitar las marcas y etiquetas del eje x
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i].reshape(28,28), cmap=plt.cm.binary) # cmap es para elegir el mapa de 
                                                                        # color, en este caso escala de grises
            plt.xlabel(train_labels[i])
        plt.show()

        ##lets build the model
        model = self.build_model()


        ##lets train the model
        history = model.fit(train_images, train_labels, epochs=iter) # Un epoch es un recorrido completo de los
                                                                # datos, si iter es 2, va a hacer el recorrido
                                                                # 2 veces, fit es para entrenar el modelo


        ##lets show the training history
        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse') # Grafica el error cuadratico medio en color rojo
        plt.show()


        ##lets show the acuraccy of the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Model accuracy = {test_acc}')

    
    def build_model(self):
        model = keras.Sequential() # Modelo secuencial en el que las capas se apilan de forma secuencial

        ##########CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)
        ##in the first layer, we use the nearest multiple of the image size from (2, 4, 8, 16, 32, 64, .....)
        ##Lets add 32 convolutional filters with 3x3 kernel
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1))) # 32 filtros o kernels
                                                                                # input_shape=(28,28,1) se espera 
                                                                                # que sean imagenes de 28x28 
                                                                                # pixeles y escala de grises 
                                                                                # (1 canal de color)

        ##is normal to add a second layer of convolutional filters the double of the size
        ##Lets add 64 convolutional filters with 3x3 kernel
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

        ##for feature extraction we use what is called pooling
        ##lets extract the features via pooling (pool size determines a matrix size for feature extraction) 
        # 2x2 = 4 features
        model.add(MaxPooling2D(pool_size=(2,2))) # Area de agrupamiento maximo, como un embudo

        ##lets randomly turn on and off neurons to improve generalization (0-1) (0-100%)
        model.add(Dropout(0.25))

        ##lets flatten the information so we can feed it to a normal neural network
        model.add(Flatten())

        ##########CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)

        ##its normal to put the number of neurons the double size of the biggest convolutional filter
        model.add(Dense(128, activation='relu'))

        ##lets add another layer for the categories we have 10 categories so we add 10 neurons
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mse'])

        return model


