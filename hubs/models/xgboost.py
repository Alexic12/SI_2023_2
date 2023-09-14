##lets import the basic libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##lets import the neural model libraries
import xgboost as xg

class xgb:
    def __init__(self, depth):
        self.depth = depth ##depth of decision tree


    def run(self, train_features, test_features, train_labels, test_labels, iter, alfa, stop_condition):
        ##lets build the model
        ##number of inputs  (for example 13 inputs, i have a depth of 10 n_estimators will be (inputs+1)*depth)
        model = self.build_model((train_features.shape[1]+1)*self.depth, alfa, 1)

        ##lets create an evaluation set
        eval_set = [(train_features, train_labels),(test_features, test_labels)]

        ##lets train the model
        model.fit(train_features, train_labels, eval_metric='mae', eval_set=eval_set, verbose=True)

        ##lets plot results
        history = model.evals_result()

        ##print(history)
        train_hist = history['validation_0']['mae']

        plt.figure()
        plt.plot(train_hist, 'r', label='Training Loss Function')
        plt.xlabel('Epoch')
        plt.ylabel('mae')
        plt.title('Training History')
        plt.legend()
        plt.show()




    def build_model(self, n_estimators, learning_rate, verbosity):
        model = xg.XGBRegressor(
            objective='reg:squarederror', ##Loss funcion for training determination
            colsample_bytree=0.5, ##Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)(between 0 - 1)
            learning_rate=learning_rate, 
            n_estimators=n_estimators, ##number of ramifications or branches (neurons)
            reg_lambda=2, ##Makes 2 cuts in the information path to force the training of the whole neural network thus, preventing overfitting
            gamma=0, ##reduces random value for reg_lambda cuts (0-1)
            max_depth=self.depth, ##Number of layers
            verbosity=verbosity, ##Shows debug info in terminal (0 None, 1 Shows info)
            subsample=0.8, ##Randomly splits the data for training for each iteration (0,1)(0-100%)
            seed=20, ##Seed for random value, for reproductibility 
            tree_method='hist', ##ramification methos (Hist: reduces significantly the amount of data to be processed)
            updater='grow_quantile_histmaker,prune'
        )

        return model


