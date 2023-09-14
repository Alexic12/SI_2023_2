import numpy as np 
import pandas as pd 
import sys 
import os 
import matplotlib.pyplot as plt
import xgboost as xg

class xgb:
    def __init__(self,depth):
        self.depth = depth
        
    
    def run(self,train_features, test_features, train_labels, test_labels, iter, alpha, stop_condition):
        model = self.build_model((train_features.shape[1]+1)*self.depth,alpha,1)
        
        eval_set = [(train_features,train_labels),(test_features,test_labels)]
        
        model.fit(train_features, train_labels, eval_metric = 'mae', eval_set = eval_set, verbose = True)
        
        history = model.evals_result()
        
        #print(history)
        
        train_hist = history['validation_0']['mae']
        
        plt.figure()
        plt.plot(train_hist,'r',label = 'Training loss function')
        plt.xlabel('Epoch')
        plt.ylabel('mae')
        plt.title('Training history')
        plt.legend()
        plt.show()
    
    def build_model(self,n_estimadores,learning_rate,verbosity):
        model = xg.XGBRegressor(
            objective='reg:squarederror', #loss function for training determination
            colsample_bytree = 0.5, #proportion of features that are randomly each 
            learning_rate = learning_rate,
            n_estimators = n_estimadores,
            reg_lambda = 2,
            gamma = 0,
            max_depth = self.depth,
            verbosity = verbosity,
            subsample = 0.8,
            seed = 20,
            tree_method = 'hist',
            updater = 'grow_quantile_histmaker,prune'  
        )
        
        return model
        