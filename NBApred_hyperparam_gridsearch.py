#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd
import time

import NBApredFuncs as pf

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#Load the pre-processed data for each n value
data_all_years = pd.read_csv("pre-processedData_n3.csv")

data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
model_data_df_n3 = pd.concat([data2015_df, data2016_df])

data_all_years = pd.read_csv("pre-processedData_n5.csv")

data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
model_data_df_n5 = pd.concat([data2015_df, data2016_df])

data_all_years = pd.read_csv("pre-processedData_n8.csv")

data2015_df = data_all_years.loc[data_all_years['Season'] == 3]
data2016_df = data_all_years.loc[data_all_years['Season'] == 4]
model_data_df_n8 = pd.concat([data2015_df, data2016_df])

#Combine the n-value dfs into one df
model_data_df = pd.concat([model_data_df_n3, model_data_df_n5, model_data_df_n8])

#validation_data_df = data_all_years.loc[data_all_years['Season'] == 5]

#Separate training and testing set
training_df = model_data_df.sample(frac=0.4, random_state=1)
indlist=list(training_df.index.values)

testing_df = model_data_df.copy().drop(index=indlist)

#Set the control switches for the layered model
plot_gains = False
fixed_wager = True
wager_pct = 0.1
classifier_model = 'GB'

#Select the classification features (eventually going to make a grid to test these)
away_features = ['teamDayOff','teamPTS', 'teamAST','teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA','teamFG%', 'team2PA','team2P%', 'team3PA','team3P%','teamFTA','teamFT%','teamORB','teamDRB','teamTREB%','teamTS%','teamEFG%','teamOREB%','teamDREB%','teamTO%','teamSTL%','teamBLKR','teamPPS','teamFIC','teamOrtg','teamDrtg','teamEDiff','teamPlay%','teamAR','teamAST/TO','teamSTL/TO']
home_features = ['opptDayOff','opptPTS','opptAST','opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA','opptFG%', 'oppt2PA','oppt2P%', 'oppt3PA','oppt3P%','opptFTA','opptFT%','opptORB','opptDRB','opptTREB%','opptTS%','opptEFG%','opptOREB%','opptDREB%','opptTO%','opptSTL%','opptBLKR','opptPPS','opptFIC','opptOrtg','opptDrtg','opptEDiff','opptPlay%','opptAR','opptAST/TO','opptSTL/TO']

class_features = ['teamAbbr','opptAbbr','Season'] + away_features + home_features
output_label = ['teamRslt'] 
#Note teamRslt = Win means visitors win, teamRslt = Loss means home wins

#Select the regression features
reg_features = ['V ML', 'H ML', 'Pred Probs V', 'Pred Probs H']
reg_label = ['Classifier Profit']

#Define the hyperparameter search grid
n_estimators_class_grid = [10, 100, 500, 1000]
learning_rate_class_grid = [0.01, 0.05, 0.1, 0.2]
max_depth_class_grid = [1, 3, 5, 10]

num_classGrid = len(n_estimators_class_grid)*len(learning_rate_class_grid)*len(max_depth_class_grid)

n_estimators_reg_grid = n_estimators_class_grid
max_depth_reg_grid = max_depth_class_grid

num_regGrid = len(n_estimators_reg_grid)*len(max_depth_reg_grid)

reg_threshold_grid = 0
n_games_avg = [3, 5, 8]
num_layerGrid = len(n_games_avg)


numGridPts = num_classGrid*num_regGrid*num_layerGrid

#Evaluate the layered model at each point on the grid
params_store = []
profit_store = []
#feature_importance_store = []

pts_tested = 0
print('Starting Grid Search')
t1 = time.time()
for c1 in n_estimators_class_grid:
    for c2 in learning_rate_class_grid:
        for c3 in max_depth_class_grid:
            for r1 in n_estimators_reg_grid:
                for r2 in max_depth_reg_grid:
                    for n1 in n_games_avg:
                        #for t1 in reg_threshold:
                    
                        pts_tested = pts_tested + 1
                    
                        class_params = [c1, c2, c3] 
                        reg_params = [r1, r2]   
                        reg_threshold = reg_threshold_grid
                        params = [c1, c2, c3, r1, r2, n1, reg_threshold]
                        
                        training_df_n = training_df.loc[training_df['Num Games Avg'] == n1]
                        testing_df_n = testing_df.loc[testing_df['Num Games Avg'] == n1]

                        #Train and test the layered model
                        class_model, classifier_feature_importance, profit_reg_model, testing_gains_reg = pf.layered_model_TrainTest(training_df_n, testing_df_n, class_features, output_label, classifier_model, class_params, reg_features, reg_label, reg_params, reg_threshold, plot_gains, fixed_wager, wager_pct)
                        model_profit = sum(testing_gains_reg)

                        #Will want to store the gridpoint params and the profit obtained 
                        #by the gridpoint model on the testing data
                        params_store.append(params)
                        profit_store.append(model_profit)
                        
                        #Maybe also store the feature importances, but that would be expensive
                        #feature_importance_store.append(classifier_feature_importance)

                        if pts_tested%10 == 0:
                            print('Grid Point', pts_tested, 'of', numGridPts)

t2 = time.time()
runtime = round((t2-t1)/60,1)
print('Done Grid Search. Run-Time:', runtime,'minutes')

#Store the grid point parameters and profits in a dataframe and save it to a csv
gridSearch_data_dict = {'Profit':profit_store}
gridSearch_data_dict['Params'] = params_store
gridSearch_df = pd.DataFrame(data=gridSearch_data_dict)
gridSearch_df.to_csv('hyperparam_gridSearch_coarse1.csv', index=False)
