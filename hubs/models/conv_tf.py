##lets import libraries 
import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense
##common libraries 
import numpy as np 
import pandas as pd 
import os 
import sys 
##import tools for data visualization 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
##import the metric libraries 
from sklearn.metrics import accuracy_score as acs
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


class conv_tf:
    def __init__(self):
        pass
    
    def run(self, train_images, test_images, train_labels, test_labels, iter):
        
        #reshape images to specify that it's a single channel
        
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        
        ##lets normalize the images
        
        train_images = train_images/255.0
        test_images = test_images/255.0
        
        #lets graph some images as example
        
        plt.figure(figsize=(10,2))
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(train_images[i].reshape(28,28), cmap=plt.cm.binary)
            plt.xlabel(train_labels[i])
        plt.show()
        
        #lets build the model 
        
        model = self.build_model()
        
        #lets train the model 
        
        history = model.fit(train_images, train_labels, epochs=iter)
        
        #lets show the training history
        
        training_data = pd.DataFrame(history.history)
        plt.figure()
        plt.plot(training_data['mse'], 'r', label = 'mse')
        plt.show()
        
        
        
        
    def build_model(self):
        model = keras.Sequential()
        
        #in the first layer, we use the nearest multiple of the image size from (2, 4, 8, 16, )
        #lets add a convolutional filter with 3x3 kernel
        
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
        
        #is normal to add a secon layer of convolutional filters the double of the size
        #lets add 64 convolutional
        
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        
        #for feature extraction we use what is called pooling
        #lets extract the features via Pooling (pool size determines a matriz size for feature extraction 2x2)
        
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #lets randomly tur on and off neurons to improve generalization (0-1) (0-100)
        
        model.add(Dropout(0,25))
        
        #lets flatten the information so we can feed it to a normal neural  network
        
        model.add(Flatten())
        
        
        
        model.add(Dense(128, activation='relu'))
        
        model.add(Dense(10, activation='softmax'))
        
        model.compile(optimizer= tf.keras.optimizers.Adam(), loss = 'mse', metrics = ['mse'])
        
        return model
        
        