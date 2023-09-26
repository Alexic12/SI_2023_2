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
    def run(self,train_images, test_images, train_labels, test_labels,iter):

        ##Reshape images to specify that its a single channel
        train_images = train_images.reshape(train_images.shape[0],28,28,1)
        test_images = test_images.reshape(test_images.shape[0],28,28,1)

        ##lets normalize the image
        train_images = train_images/255.0
        test_images = test_images/255.0

        ##lets graph some images as example

        plt.figure(figsize=(10,2))

        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(train_images[i].reshape(28,28), cmap = plt.cm.binary)
            plt.xlabel(train_labels[i])
        plt.show()

        model =self.build_model()

        history= model.fit(train_images,train_labels,epochs=iter)
        
        
        training_data = pd.DataFrame(history.history)
        #print(training_data)

        plt.figure()
        plt.plot(training_data['mse'], 'r', label='mse')
        plt.show()

        ##Accuracy

        test_loss, test_acc= model.evaluate(test_images,test_labels)
        print(f"Model accuracy= {test_acc}")

        
        def build_model(self):
            model = keras.sequential()

            ##CONVOLUTIONAL LAYERS

            ##CONVOLUTIONAL FILTER 3 x 3
            model.add(Conv2D(32, kernel_size= (3,3), activation="relu", input_shape=(28,28,1)))


            model.add(Conv2D(64, kernel_size= (3,3), activation="relu"))

            ##for feature extraction we use pooling
            ##extract the features  pool size refers the max extraction

            model.add(MaxPooling2D(pool_size=(2,2)))

            ##Randomly turn off neurons to improve generalization
            model.add(Dropout(0.25))

            ##flattening the information so we can feed it to a normal neural network

            model.add(Flatten())
            ##IMAGE FILTERING

            ##NUMBER OF NEURONS IS DOUBLE CONVOLUTIONAL FILTER
            model.add(Dense(128, activation="relu"))

            ##another layer for categories
            model.add(Dense(10, activation="softmax"))

            model.compile(optimizer= tf.keras.optimizers.Adam(), loss="mse", metrics= ["mse"])
            
            return model
            
