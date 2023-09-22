import numpy as np 
import pandas as pd 
import sys 
import os 
import matplotlib.pyplot as plt
import xgboost as xg
from sklearn.preprocessing import StandardScaler
##import the metric libraries 
from sklearn.metrics import accuracy_score as acs

class xgb:
    def __init__(self,depth):
        self.depth = depth
        
    
    def run(self,train_features, test_features, train_labels, test_labels, original_feature,iter, alpha, stop_condition, chk_name, train):
        model = self.build_model((train_features.shape[1]+1)*self.depth,alpha,1)
        
        eval_set = [(train_features,train_labels),(test_features,test_labels)]
        
        if train:
        
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
        
            ##Validation step 
            
            pred_out = model.predict(test_features)
            ##scaler = StandardScaler()
            ##data_labels_norm = scaler.fit_transform(pred_out)
            
            plt.figure()
            plt.plot(pred_out, 'r', label='Prediction output')
            plt.plot(test_labels, 'b', label='Real output')
            plt.xlabel('Data points')
            plt.ylabel('normalize value')
            plt.title('Validation')
            plt.show()
            
            #accuracy metric 
            accurancy = acs(test_labels.astype(int),pred_out.astype(int))*100
            
            print(f'Accurancy: {accurancy:.2f}%')
            
            ##lets ask if the user wants to store the model
            
            r = input('Save model? : (Y-N)') 
            if r == 'Y':
                
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
                checkpoint_file = os.path.join(model_dir, f'{chk_name}.json')
                print(f'checkpoint path: {checkpoint_file}')
                model.save_model(checkpoint_file)
                print('Model saved!')
                
            elif r == 'N':
                
                print('Model NOT saved!')
                
            else:
                print('Command not recognized')
                
        else:
            ##we are not training a model here, just using a already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
            checkpoint_file = os.path.join(model_dir, f'{chk_name}.json') 
            ##lets load the model
            model.load_model(checkpoint_file)
            
            ##Prediction output 
            pred_out = model.predict(train_features)
            
            data = pd.DataFrame(train_features)
            
            print(f'Dataframe: {data}')
            ##scaler = StandardScaler()
            ##data_labels_norm = scaler.fit_transform(pred_out)
            
            plt.figure()
            plt.plot(pred_out, 'r', label='Prediction output')
            plt.plot(train_labels, 'b', label='Real output')
            plt.xlabel('Data points')
            plt.ylabel('normalize value')
            plt.title(f'Predict output of model {chk_name}')
            plt.show()
            
            #accuracy metric 
            accurancy = acs(train_labels.astype(int),pred_out.astype(int))*100
            
            print(f'Prediction Accurancy: {accurancy:.2f}%')
            
    
    def build_model(self,n_estimadores,learning_rate,verbosity):
        model = xg.XGBRegressor(
            objective='reg:squarederror', #loss function for training determination
            colsample_bytree = 0.5, ##Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)(between 0 - 1)
            learning_rate = learning_rate,
            n_estimators = n_estimadores, ##number of ramifications or branches (neurons)
            reg_lambda = 2, ##Makes 2 cuts in the information path to force the training of the whole neural network thus, preventing overfitting
            gamma = 0, ##reduces random value for reg_lambda cuts (0-1)
            max_depth = self.depth, ##Number of layers
            verbosity = verbosity, ##Shows debug info in terminal (0 None, 1 Shows info)
            subsample = 0.8, ##Randomly splits the data for training for each iteration (0,1)(0-100%)
            seed = 20, ##Seed for random value, for reproductibility
            tree_method = 'hist', ##ramification methos (Hist: reduces significantly the amount of data to be processed)
            updater = 'grow_quantile_histmaker,prune'  
        )
        
        return model
        