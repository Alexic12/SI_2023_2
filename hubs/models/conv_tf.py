##lets import libraries
import tensorflow as tf 
from tensorflow import keras 
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten

#common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for visualization
import matplotlib.pyplot as plt

##Import the metrics libraries 
from sklearn.metrics import accuracy_score as acs

class conv_tf:

    def __init__(self):
        pass

    def run(self,train_images,test_images,train_labels,test_labels,iter):

        ##Reshape images to specify that it's a single channel
        train_images = train_images.reshape(train_images.shape[0],28,28,1)
        test_images = test_images.reshape(test_images.shape[0],28,28,1)

        ##Let's normalize the image 
        train_images = train_images/255.0
        test_images = test_images/255.0

        ##Let's graph some images as example
        plt.figure(figsize=(10,2))

        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i].reshape(28,28),cmap=plt.cm.binary)
            plt.xlabel(train_labels[i])
        plt.show()

        ##let's build the model
        model = self.build_model()

        ##let's train the model
        history = model.fit(train_images,train_labels,epochs=iter)

        ##Let's show the training history
        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'], 'r', label = 'mse')
        plt.show()

        ##Let's show the accuracy of the model
        test_loss,test_acc = model.evaluate(test_images,test_labels)
        print(f'Model Accuracy: {test_acc}')

    def build_model(self):
        model = keras.Sequential()
        
        ##########CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)############
        ##In the first layer, we use the nearest multiple of the image size from (2,4,8,16,32,64,.....)
        ##Let's add 32 convolutional filter with 3x3 kernel
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (28,28,1)))

        ##It's common to add a second layer of convolutional filters with double the size
        ##Let's add 4 convolutional filters with 3x3 kernel
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

        ##for feature extraction we use a method called pooling
        ##Let's extract the features via pooling (pool size determines a matrix size for feature extraction) 2x2 = 4 characteristics
        model.add(MaxPooling2D(pool_size = (2,2)))

        ##Let's randomly turn on and off neurons to improve generalization (0-1) (0-100%)
        model.add(Dropout(0.25))

        ##Let's flatten the information in order to feed it to a normal neural network
        model.add(Flatten())

        ##########CONVOLUTIONAL LAYERS (IMAGE FILTERING AND FEATURE EXTRACTION)############

        ##it's common to put the number of neurons as double the size of the biggest convolutional filter
        model.add(Dense(128,activation ='relu'))

        ##Let's add another layer for the categories we have (10 categories so we add 10 neurons) 
        model.add(Dense(10,activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'mse', metrics=['mse'])

        return model
    
