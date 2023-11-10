##lets import libraries
import xgboost as xg

#common modules
import numpy as np
import pandas as pd
import os
import sys

##import tools for visualization
import matplotlib.pyplot as plt

##Import the metrics libraries 
from sklearn.metrics import accuracy_score as acs

##For denormalizing the data
from sklearn.preprocessing import StandardScaler


class xgb:

    def __init__(self,depth):
        self.depth = depth ##depth of decision tree

    def run(self,train_features, test_features, train_labels, test_labels,original_features,original_labels,iter,alfa,stop_condition,chk_name,train,neurons):
        ##let's build the model
        ##number of inputs (for example 13 inputs, i have a depth of 10, n_estimators will be (inputs+1)*depth)
        model = self.build_model((train_features.shape[1]+1)*self.depth, alfa, 1 )

        ##let's create an evaluation set
        eval_set = [(train_features,train_labels),(test_features,test_labels)]

        ##let's train the model
        if train:
            model.fit(train_features, train_labels, eval_metric = 'mae', eval_set = eval_set, verbose = True)

            ##Let's a feature importance weight analysis
            self.run_weight_analysis(model)

            ##Let's plot the result
            history = model.evals_result()

            #print(history)
            train_hist = history['validation_0']['mae']
            
            plt.figure()
            plt.plot(train_hist,'r', label = 'Training Loss Function')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Training History')
            plt.legend()
            plt.show()

            ##validation step
            pred_out = model.predict(test_features)
            plt.figure()
            plt.plot(pred_out, 'r', label='Model Output')
            plt.plot(test_labels,'b', label = 'Real Output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title('Validation')
            plt.legend()
            plt.show()

            ##Let's show the accuracy value for this training batch
            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Accuracy: {accuracy:.2f}%')

            ##Let's ask if the user wants to store the model
            r = input("Save model? (Y-N)")
            if r == 'Y':
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
                chekpoint_file = os.path.join(model_dir, f'{chk_name}.json')
                print(f'Checkpoint path: {chekpoint_file}')
                model.save_model(chekpoint_file)
                print('Model Saved')

            elif r == 'N':
                print('Model NOT Saved')

            else:
                print('Command not recognized')
        
        else:
            ##We are not training a model here, just using an already existing model
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
            chekpoint_file = os.path.join(model_dir, f'{chk_name}.json')

            ##Let's load the model
            model.load_model(chekpoint_file)

            ##Predicition Output
            pred_out = model.predict(train_features)

            ##Let's denormalize the data
            SC = StandardScaler()

            original_labels_norm = SC.fit_transform(original_labels)

            if neurons == 1:
                pred_out = pred_out.reshape(-1,1)

            pred_out_denorm = SC.inverse_transform(pred_out)

            pred_df = pd.DataFrame(pred_out_denorm)

            result_data = pd.concat([original_features,pred_df],axis=1)

            print(f'Dataframe : {result_data}')
            
            results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data','results'))

            results_file = os.path.join(results_dir, f'{chk_name}_RESULTS_XGB.xlsx')

            ##Let's store the dataframe as excel file
            result_data.to_excel(results_file, index = False, engine = 'openpyxl')

            plt.figure()
            plt.plot(pred_out, 'r', label='Model Output')
            plt.plot(train_labels,'b', label = 'Real Output')
            plt.xlabel('data points')
            plt.ylabel('Normalized value')
            plt.title(f'Prediction Output of model {chk_name}')
            plt.legend()
            plt.show()

            ##Let's show the accuracy value for this training batch
            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction Accuracy: {accuracy:.2f}%')

    def build_model(self, n_estimators, learning_rate, verbosity):
        model = xg.XGBRegressor(
            objective = 'reg:squarederror', ##Loss function for training determination
            colsample_bytree = 0.5, ## Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)(between 0 and 1)
            learning_rate = learning_rate,
            n_estimators = n_estimators, ##Number of ramifications or branches (neurons)
            reg_lambda = 2, ##Creates 2 cuts in the information path to force the training of the whole neural network thus, preventing overfitting
            gamma = 0, ##reduces random value for reg_lambda cuts (0-1)(0 is random)
            max_depth = self.depth, ##Number of Layers
            verbosity = verbosity, ##Shows debug info in terminal (0 None, 1 shows info)
            subsample = 0.8, ##Randomly splits the data for training for each iteration(0,1)(0-100%)
            seed = 20, ##Seed for random value, for reproductibility
            tree_method = 'hist', #ramification methods( Hist, significantly reduces the amount of data to be processed)
            updater =  'grow_quantile_histmaker,prune'
        )
        return model

    def run_weight_analysis(self,model):
        feature_importance = model.feature_importances_
        print("Feature Importance")
        print(feature_importance)
        plt.figure(figsize=(10,5))
        xg.plot_importance(model,importance_type = "weight")
        plt.show()
    
    def load_model(self,name,inputs,alfa):
        model = self.build_model((inputs+1)*self.depth, alfa, 1)
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
        model_file = os.path.join(model_dir, f'{name}.json')
        print(f'Path : {model_file}')
        model.load_model(model_file)

        return model

    


        
        


    