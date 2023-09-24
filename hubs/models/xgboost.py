import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##NEURAL MODEL LIBRARIES
import xgboost as xg

##Accuracy
from sklearn.metrics import accuracy_score as acs

class xgb:
    def __init__(self,depth):
        self.depth = depth       
        
    def run(self, train_features, test_features, train_labels, test_labels, original_features, iter, alfa, stop_condition,chk_name,train):
        model = self.build_model((train_features.shape[1]+1)*selef.depth, self.depth, alfa, 1)
        ##Evaluation set
        eval_set = [(train_features, train_labels),(test_features, test_labels)]

        if train: 
            ##training the model
            model.fit(train_features, train_labels, eval_metric = "mae", eval_set = eval_set, verbose = True)
            ##Plot results
            history = model.evals_result()

            ##print(history)
            train_hist = history["validation_0"]["mae"]

            plt.figure()
            plt.plot(train_hist, "r", label = "Training loss function")
            plt.xlabel("Epoch")
            plt.ylabel("mae")
            plt.title("Training history")
            plt.legend()
            plt.show()

            ##validation step
            pred_out = model.predict(train_features)

            plt.figure()
            plt.plot(pred_out, "r", label="Model Output")
            plt.plot(test_labels, "b", label="Real Output")
            plt.xlabel("data points")
            plt.ylabel("Validation")
            plt.title("Validation")
            plt.legend()
            plt.show()

            ##Accuracy

            accuracy = acs(test_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Accuracy: {accuracy:.2f}%')
            
            ##Asking if the user wants to store the model
            r = input("Save Model? : ")
            if r == "Y":
                model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
                checkpoint_file = os.path.join(model_dir, f"{chk_name}.json")
                model.save_model(checkpoint_file)
            elif r == "N":
                print("Model NOT Saved")

            else:
                print("Command not recognized")
        else:
            ##NOT TRAINING , USING AN ALREADY EXISTING MODEL, MUST BE EQUAL DIMENSION
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', 'xgboost'))
            checkpoint_file = os.path.join(model_dir, f"{chk_name}.json")
            model.load_model(checkpoint_file)

            pred_out = model.predict(train_features)

            plt.figure()
            plt.plot(pred_out, "r", label="Model Output")
            plt.plot(train_labels, "b", label="Real Output")
            plt.xlabel("data points")
            plt.ylabel("Validation")
            plt.title("Prediction Output of model {chk_name}")
            plt.legend()
            plt.show()
            
            ##Accuracy

            accuracy = acs(train_labels.astype(int), pred_out.astype(int)) * 100
            print(f'Prediction Accuracy: {accuracy:.2f}%')
            

    def build_model(self, n_estimators, learning_rate, verbosity):
        model = xg.XGBRegressor(
            objective = "reg:squarederror", ##Loss function for training determination
            colsample_bytree = 0.5, ##Proportion of features that are randomly sampled each iteration(reducing this parameter can prevent overfitting)
            learning_rate = learning_rate,
            n_estimators = n_estimators, ##Number of ramification or branches(neurons)
            reg_lambda = 2, ##Makes cuts in the information path to force the training of the whole neural network
            gamma = 0, ##Reduces random value for reg_lamba cuts
            max_depth = self.depth, ##Number of layers
            verbosity = verbosity,
            subsample = 0.8, ##Randomly splits the data for training for each iteration
            seed = 20, ##Seed for random value, for reproductibility
            tree_method = "hist", ##Ramification method, hist reduces significantly the amount of data to be processed
            updater = "grow_quantile_histmaker,prune"

        )

        return model


        




