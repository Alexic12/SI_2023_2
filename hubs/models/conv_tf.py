#Lets import libraries
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import numpy as np
import pandas as pd
import os
import sys

#Import tools for data visualization
import matplotlib.pyplot as plt

#Import the metrics libraries
from sklearn.metrics import accuracy_score as acs

##for denormalizing the data
from sklearn.preprocessing import StandardScaler

class conv_tf:
    def __init__(self):
        pass

    def run(self, train_images, test_images, train_labels, test_labels, iter):
        
        #Reshape images to specify that it's a single channel
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) #28x28 pixeles, y 1 solo color
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

        #Lets normalize the image
        train_images = train_images/255.0
        test_images = test_images/255.0

        #Lets graph some images
        plt.figure(figsize=(10,2))

        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
            plt.xlabel(train_labels[i])
        plt.show()

        #lets build the model
        model = self.build_model()

        #Lets trains the model
        history=model.fit(train_images, train_labels, epochs=iter)

        #Lets show the training history
        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()

        #Lets show the acuraccy of the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Model accuracy= {test_acc}')


        def build_model(self):
            model = keras.sequential()

            #############CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)#################
            #In the first layer, we use the nearest multiple of the image size from (2, 4, 8, 16, 32, 64, ....)
            #Lets add 32 convolutional filter with 3x3
            model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(28,28,1)))

            #Is normal to add a second layer of convolutional filters the double of size
            #Lets add 64 convolutional filters with 3x3 kernel
            model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

            #
            #Lets extract the features
            model.add(MaxPooling2D(pool_size=(2,2)))

            #Lets randomly turn on and off neurons to improve generalization (0-1) (0-100%)
            model.add(Dropout(0.25))

            #Lets flatten the information so we can feed it to a normal neural network
            model.add(Flatten())

            #############CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)#################

            #Its normal to put the number of layers
            model.add(Dense(128, activation='relu'))

            #Lets add another layer for the categories we have 10 categories so we add 10 neurons
            model.add(Dense(10, activation='softmax'))

            model.compiler(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mse'])

            return model
